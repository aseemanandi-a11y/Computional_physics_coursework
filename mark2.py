#CURRENT PROBLEM: SODS SHOCK TUBE USING NAVIER STOKES EQUATIONS
# for sods shock tube the boundary conditions are transmissive i.e. the ghost cells will occupy the nearest cell
#has not accounted for Fv or the diffusion timestep
# numerical diffusion introduced by rusanov counteracted by sufficiently high resolution

# IMPORTS AND CONSTANTS
import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4      # ideal gas ratio
xs = 0.5         # discontinuity point for sods experiment
R = 1            # specific gas constant J/kg/K using dimensionless parameters
mu = 1.8*10**(-5)# dynamic viscosity Pa-s
k = 0.026        # thermal conductivity W/m/K
L = 1.0          # since length is not a relevant quantity to the solution of the navier stokes the legth of the grid is assumed to be 1 unit
tfinal = 0.2     # the end time for the loop

# variable 
N = 2400          # number of cells variable with resolution
Re = 100.0       # Reynolds number
Pr = 0.7         # Prandts number

# calculated from above
dr = L / N              # the distance between two points 
mu = 1.0 / Re           # dimensionless dynamic viscosity
k = 3.5 / (Re*Pr)       # dimensionless thermal diffusivity
cp = gamma*R/(gamma - 1)#specific heat capacity under constant pressure

# FUNCTIONS

#reinforces transmissive boundary condition
def reinforce(U):
    U[:2] = U[2]                     
    U[2+N:] = U[N+1]
    return U

#calculates convective flux from current conserved variables
def convective(U):

    Fc = np.zeros_like(U)     #creating the flux vector

    rho = U[:, 0]     #density
    rhov = U[:, 1]    #momentum
    E = U[:, 2]       #Energy

    v = rhov / rho     #velocity
    p = (gamma - 1) * (E - 0.5*rhov**2 / rho)     #the state function used to calculate pressure from conserved variables

    #updating the Fc
    Fc[:, 0] = rhov
    Fc[:, 1] = rhov*v + p
    Fc[:, 2] = v*(E + p)

    return Fc

#calculates viscous flux from the conserved variables
#!!! this one is not quite right it needs to be assigning values to interfaces no cell centres !!!
def viscous(U):

    Fv = np.zeros_like(U)     #creating the flux vector

    rho = U[:, 0]
    rhov = U[:, 1]
    E = U[:, 2]

    u = rhov / rho        #velocity of fluid
    p = (gamma - 1)*(E - 0.5*rhov**2 / rho)
    T = p / (rho * R)     #Temperature

    dudx = np.zeros_like(u)
    dTdx = np.zeros_like(T)

    dudx[1:-1] = (u[2:] - u[:-2]) / (2*dr)    #using central finite difference
    dTdx[1:-1] = (T[2:] - T[:-2]) / (2*dr)

    txx = (4/3) * mu * dudx

    Fv[:, 0] = 0                 # viscosity has no contribution to mass
    Fv[:, 1] = txx               # viscous contribution to momentum flux
    Fv[:, 2] = u*txx + k*dTdx    # term1 is energy transportation due to viscosity and term2 is energy transportation by heat diffusion

    return Fv

#Rusanovs finite volume methd
def rusanov(U,Fc,Fv):
    UL = U[:-1]     #getting left and right cells from both vectors
    UR = U[1:]

    FL = Fc[:-1]
    FR = Fc[1:]

    rhol = UL[:, 0]
    vl = UL[:, 1] / rhol
    pl = (gamma - 1)*(UL[:, 2] - 0.5*UL[:, 1]**2/rhol)

    rhor = UR[:, 0]
    vr = UR[:, 1] / rhor
    pr = (gamma - 1)*(UR[:, 2] - 0.5*UR[:, 1]**2/rhor)

    cl = np.sqrt(gamma * pl / rhol)     #gives the speed of sound in the neighbouring cells
    cr = np.sqrt(gamma * pr / rhor)

    a = np.maximum(np.abs(vl) + cl,np.abs(vr) + cr)     #gets the highest total of fluid speed +speed of sound i.e. the highest speed of sound possible

    # Rusanov numerical flux at interfaces
    Fint = 0.5*(FL + FR) - 0.5*a[:, None]*(UR - UL)     #array that contains the flux at every interface

    # Update physical cells only
    dUdt = -(Fint[2:N+2] - Fint[1:N+1]) / dr  + (Fv[2:N+2] - Fv[1:N+1]) / dr        #array that calculates the spatial difference using adjacent interfaces does not include ghost cells

    return dUdt

#takes a singlestep of the forward euler
def f_euler_singlestep(U, dt):           

    reinforce(U)
    Fc = convective(U)
    Fv = viscous(U)
    DU = rusanov(U,Fc,Fv)
    U_new = U.copy()
    U_new[2:N+2] = U[2:N+2] + dt * DU
    reinforce(U_new)

    return U_new

#performs rk2 using the euler step
def rk2(U, dt):

    Ustar = f_euler_singlestep(U, dt)
    Ue = f_euler_singlestep(Ustar, dt)
    U_new = U.copy()
    U_new[2:N+2] = (0.5 * U[2:N+2]+ 0.5 * Ue[2:N+2])
    reinforce(U_new)
    
    return U_new

def compute_dt(U, CFL=0.5, Cdiff = 0.5):
    rho = U[2:N+2, 0]
    rhov = U[2:N+2, 1]
    E = U[2:N+2, 2]

    u = rhov / rho
    p = (gamma - 1) * (E - 0.5 * rhov**2 / rho)
    c = np.sqrt(gamma * p / rho)
    nu = mu / rho
    alpha = k / (rho * cp)

    max_speed = np.max(np.abs(u) + c)

    dt_conv = CFL * dr / max_speed
    dt_diff = Cdiff * dr**2 / np.max(np.maximum(nu, alpha))

    dt = min(dt_conv, dt_diff)

    return dt

#INITIALISING THE GRID

# Physical cell centers
r = (np.arange(N) + 0.5) * dr     #the 0.5 is so that we maintain our values at the centre of the cells stores each position value between 0 and 1

#ghost zones are used to maintain boundary conditions
rghost = np.zeros(N + 4)
rghost[2:2+N] = r                     # two ghost cells on either side of the main set of cells
rghost[:2] = r[0] - dr * np.array([2, 1])     # currently using transmissive condition for sods
rghost[2+N:] = r[-1] + dr * np.array([1, 2])

#INITIAL CONDITIONS - changes based on the problem we are working on 
# we use vector U to store our conserved variables(rho,rhov,E)
# vector Fc stores the euler flux variables (rhov,rhov^2+p,u(E+p))
# vector Fv stores the viscous and heat conduction flux variables (0,Txx,uTxx-qx)

U = np.zeros((N + 4, 3))     #U[][0] = rho, U[][1] = rhov, U[][2] = E

#applying sods initial conditions
left = r<xs     #boolean mask
U[2:N+2,0][left] = 1
U[2:N+2, 1][left]  = 0
U[2:N+2, 2][left]  = 2.5

U[2:N+2, 0][~left] = 0.125
U[2:N+2, 1][~left] = 0
U[2:N+2, 2][~left] = 0.25

U = reinforce(U)     #reinforcing boundary conditions

#LOOP
t= 0
snapshot_time = [0,0.05, 0.1, 0.15, 0.2]
snapshot = []

while t < tfinal:
    
    dt = compute_dt(U)     #computes next timestep
    dt = min(dt, tfinal - t)

    future_times = [ts for ts in snapshot_time if ts > t + 1e-10]     #section to ensure the code passes through the required timestamps
    if future_times:
        dt = min(dt, min(future_times) - t)

    for ts in snapshot_time:
        if abs(t - ts) < 1e-10:
            snapshot.append(U.copy()) #does not append the final value as time evolution happens after

    U = rk2(U, dt)     #the actual evolution

    t = t + dt
snapshot.append(U.copy())     #to appendfinal snapshot

#RESULT

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, Us in enumerate(snapshot):
    Us = Us[2:N+2] 

    rho = Us[:, 0]
    rhov = Us[:, 1]
    E = Us[:, 2]

    u = rhov / rho
    p = (gamma - 1) * (E - 0.5 * rhov**2 / rho)

    t_snapshot = snapshot_time[i]

    axes[0].plot(r, rho, label=f"t = {t_snapshot:.2f}")
    axes[1].plot(r, u, label=f"t = {t_snapshot:.2f}")
    axes[2].plot(r, p, label=f"t = {t_snapshot:.2f}")

axes[0].set_ylabel(r"Density $\rho$")
axes[1].set_ylabel(r"Velocity $u$")
axes[2].set_ylabel(r"Pressure $p$")

for ax in axes:
    ax.set_xlabel("r")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()