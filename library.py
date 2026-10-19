import math

#the complex numbers structure
class MyComplex:
    def __init__(self,real,complex):
        self.real = real
        self.complex = complex

    def display(self):
        return(self.real, '+', self.complex, 'i')  #f"{self.real} + {self.imag}i"
    
    def addcmplx(self,c2):
        c = MyComplex(self.real+c2.real,self.complex+c2.complex)
        return(c)
    
    def subcmplx(self,c2):
        c = MyComplex(self.real-c2.real,self.complex-c2.complex)
        return(c)
    
    def prodcmplx(self,c2):
        c = MyComplex(self.real*c2.real - self.complex*c2.complex, self.real*c2.complex + self.complex*c2.real)
        return(c)

    def modulus(self):
       return math.sqrt(self.real**2 + self.imag**2)

#Random function: generates a random integer from a range
class random:
    def __init__(self, seed=0.1):
        self.a = 1103515245
        self.c = 12345
        self.m = 32768
        self.x0 = seed

    def next_rand(self):
        xi = ((self.a * self.x0 + self.c) % self.m) / self.m
        self.x0 = xi
        return xi


#matrix_multiply(A, B): Multiplies two matrices
def matrix_multiply(A,B):
     if len(A[0]) == len(B):
         row1 = []
         AB = []
         val = 0
         for i in range (0,len(A)):
             for k in range(0,len(B[0])):
                 for j in range(0,len(A[0])):
                     val = val + A[i][j]*B[j][k]
                 row1.append(val)
                 val = 0
             AB.append(row1)
             row1 = []
         return AB
     else:
          raise ValueError("These matrices cannot be multiplied.")
     
#matrix_vector_multiply(A, x): Multiplies a matrix by a vector
def matvec(M,v):
     if len(M[0]) != len(v):
          raise ValueError("Matrix and vector cannot be multiplied.")
     else:
         Mv = []
         val = 0
         for i in range (0,len(M)):
             for j in range(0,len(v)):
                 val = val + M[i][j]*v[j]
             Mv.append(val)
             val = 0
         return Mv

#dot_product(v1, v2): Calculates the dot product of two vectors
def dotprod(a,b):
     ip = 0
     for i in range(0,len(a)):
         ip = ip + a[i]*b[i]
     return(ip)

#solves matrix
def Gauss_jordan(A,b):

    #creating the augmented matrix
    #A_aug = [row.copy() for row in A]
    A_aug = A
    for bi in range(len(b)):
        A_aug[bi].append(b[bi])

    #column iteration begins here
    for c in range(len(A)):

        #finding and switching maximum element row
        searchmax = []
        for i in range (c,len(A)):
            searchmax.append(abs(A_aug[i][c]))
        pivot = searchmax.index(max(searchmax)) + c
        A_aug[c],A_aug[pivot] = A_aug[pivot],A_aug[c]

        #normalising row
        div = A_aug[c][c]
        for i in range(len(A_aug[c])):
            element = A_aug[c][i]
            element = element/div
            A_aug[c][i] = element

        #getting zeroes in the rest of the column below chosen value
        for r in range(len(A)):
            if r != c:
                factor = A_aug[r][c]/A_aug[c][c]
                for element in range(len(A_aug[r])):
                    A_aug[r][element] = A_aug[r][element] - factor*A_aug[c][element]
    #output
    l = []
    for i in range(len(A)):
        l.append(A_aug[i][len(A)])

    return l

def LU(A):

    for k in range(len(A)):
        for i in range(k + 1, len(A)):
            A[i][k] = A[i][k] / A[k][k]
            
            for j in range(k + 1, len(A)):
                A[i][j] = A[i][j] - A[i][k] * A[k][j]
                
    return A

#decomposes into L and L transpose
def cholesky(A):
    #check if matrix is symmetric
    for i in range (len(A)):
        for j in range(len(A)):
            if A[i][j] != A[j][i]:
                return 'matrix is not symmetric'
    for i in range(len(A)):
        for j in range(i+1):
            sum = 0
            for k in range (j):
                sum = sum + (A[i][k]*A[j][k])
            if i == j:
                A[i][j] = math.sqrt(A[i][j] - sum)
            else:
                A[i][j] = (A[i][j] - sum)/(A[j][j])
                A[j][i] = A[i][j]

    return A

#uses LU and Cholesky to solve
def forback(A,b,LU = True):
    for i in range (len(A)):
        for j in range(len(A)):
            if A[i][j] != A[j][i]:
                LU = True
            else:
                LU = False
    y = []
    # forward substitution
    for i in range(len(A)):
        sum = 0
        for j in range(i):
            sum = sum + A[i][j]*y[j]
        if LU == True:
            y.append(b[i] - sum)
        else:
            y.append((b[i] - sum)/A[i][i])


    x = [0.0] * len(A)

    #backward substitution
    for i in range(len(A) - 1,-1,-1):
        sum = 0
        for j in range(i+1,len(A)):
            sum = sum + A[i][j]*x[j]
        x[i] = (y[i] - sum)/A[i][i]

    return x

#different way of solving using a guess and iterating formula
def jacobi(A,b,tol = 1e-6):

    #checking diagonal dominance:
    for i in range (len(A)):
        s = 0
        for j in range(len(A)):
            if j != i:
                s = s+abs(A[i][j])
        if A[i][i] <= s:
            print('the matrix is not diagonally dominant',A[i][i],s,i)
            diagonal = False
            return

    #solving the equation
    xn = [0] * len(b)
    x0 = [0] * len(b)

    for k in range(1, 501): 
        for i in range(len(b)):
            sum = 0.0
            for j in range(len(b)):
                if j != i:
                    sum = sum + A[i][j] * x0[j]

            xn[i] = (b[i] - sum) / A[i][i]

        converged = True
        for i in range(len(b)):
            if abs(xn[i] - x0[i]) >= tol:
                converged = False
                break
        if converged == True :
            return xn,k

        x0 = list(xn)
    return xn,k

#basicaqlly same as jacobi but it immediately uses new values instead of waiting for next iteration using a weight system
def SOR(A,b,tol = 1e-6,w = 1):

    #checking diagonal dominance:
    for i in range (len(A)):
        s = 0
        for j in range(len(A)):
            if j != i:
                s = s+abs(A[i][j])
        if abs(A[i][i]) <= s:
            print('the matrix is not diagonally dominant')
            diagonal = False
            return

#solving the equations
    x0 = [0] * len(b)

    for k in range(1, 501): 
        xtemp = list(x0)
        for i in range(len(b)):
            sum = 0.0
            for j in range(len(b)):
                if j != i:
                    sum = sum + A[i][j] * x0[j]

            x0[i] = (1 - w) * xtemp[i] + w * ((b[i] - sum) / A[i][i])

        converged = True
        for i in range(len(b)):
            if abs(x0[i] - xtemp[i]) >= tol:
                converged = False
                break
        if converged == True :
            return x0,k

        x0 = x0
    return x0,k

#finds root of function using intermediate value theorem fa and fb must be on opposite sides of 0
def bisection(f,a,b,delta = 1e-6):

    #checking unexpected cases
    if f(a)*f(b) > 0:
        print('no zero in bracket')
        return None
    if f(a)*f(b) == 0:
        if f(a) == 0:
            return a,0
        else:
            return b,0

    #actual method for 1000 max iterations
    for i in range(100):
        c = (a + b)/2
        if abs(f(c)) <= delta or (b - a)/2 <=delta:
            return c,i
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    print('zero not found within 100 iterations try different bracket values')
    return None

#finds root using intersection of secant line with x axis
def regfals(f,a,b,delta = 1e-6):

    #checking unexpected cases
    if f(a)*f(b) > 0:
        print('no zero in bracket')
        return None
    if f(a)*f(b) == 0:
        if f(a) == 0:
            return a,0
        else:
            return b,0

    c0 = a

    #actual method for 100 max iterations
    for i in range(100):
        c = b - ((b - a)*f(b)/(f(b) - f(a)))
        if abs(f(c)) <= delta or abs(c-c0)<=delta:
            return c,i
        c0 = c
        if f(a)*f(c) < 0:
            b = c
        else:
            a = c
    print('zero not found within 100 iterations try different bracket values')
    return None

#finds 0 using tangent at x
def newton_raphson(f,df,x0,delta = 1e-6):

    x = x0

    #starting iterations with maximum 1000 iterations.
    for i in range(1000):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-6:
            print("Derivative too close to zero. Method fails.")
            return None
        x1 = x - (fx/dfx)
        if abs(x1 - x) < delta:
            return x1,i
        x = x1
    print('method did not converge after 1000 iterations')
    return x,1000

#solves using x = g(x) instead of f(x) = 0
def fixed_point(g,x0,delta=1e-6):
    for i in range(1000):
        x1 = g(x0)
        if abs(x1-x0) < delta:
            return x1,i
        x0 = x1
    print('method did not converge after 1000 iterations')

    #import math

#evaluates first and second derivatives of a given polynomial
def derivatives(coeffs,x0):
    p = 0
    dp = 0
    ddp = 0
    for c in coeffs:
        ddp = ddp * x0 + 2 * dp
        dp = dp * x0 + p
        p = p * x0 + c
        
    return p, dp, ddp
    

def laguerre_method(coeffs, x0, tol=1e-12):
    n = len(coeffs) - 1
    if n <= 0:
        return None
        
    x = x0
    
    for i in range(100):
        p,dp,ddp = derivatives(coeffs,x)
        
        if abs(p) < tol:
            return x
            
        G = dp/p
        H = G**2 - ddp/p
        
        inner = (H * n - G**2) * (n - 1)
        inner_sqrt = math.sqrt(inner)

        denom1 = G + inner_sqrt
        denom2 = G - inner_sqrt
        denom = denom1 if abs(denom1) > abs(denom2) else denom2
        
        if denom == 0.0:
            a = 0
        else:
            a = n/denom
            
        # x = x - a
        x = x - a
        
        if abs(a) < tol:
            break
            
    return x

def deflate_polynomial(coeffs, root):
    deflated = []
    remainder = 0
    for c in coeffs[:-1]:
        remainder = remainder*root + c
        deflated.append(remainder)
    return deflated

def midpoint(f, a, b, n = 10):
    h = (b - a) / n
    sum = 0
    for i in range(n):
        sum = sum + f(a + (i + 0.5) * h)
    return sum * h

def trapezoidal(f, a, b, n = 10):
    h = (b - a) / n
    sum = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        sum = sum + f(a + i * h)
    return sum * h


def simpson(f, a, b, n):
  if n % 2 != 0:
    raise ValueError("n must be an even number.")
  h = (b - a) / n
  sum = f(a) + f(b)
  for i in range(1, n):
    k = a + i * h
    if i % 2 == 0:
      sum = sum + 2 * f(k)
    else:
      sum = sum + 4 * f(k)
  return sum * h / 3

def montecarlo(f, a, b, n=100000):
  rng = random(900)
  sum = 0
  for i in range(n):
    r = rng.next_rand()*(b - a) + a
    sum = sum + f(r)
  return (b - a) * (sum / n)

Tabs = {"legendre": {"nodes": [-0.9602898564975363, -0.7966664774136267, -0.5255324099163290, -0.1834346424956498, 0.1834346424956498,  0.5255324099163290,  0.7966664774136267,  0.9602898564975363],
        "weights": [ 0.1012285362903763,  0.2223810344533745,  0.3137066458778873,  0.3626837833783620, 0.3626837833783620,  0.3137066458778873,  0.2223810344533745,  0.1012285362903763]
    },
    "laguerre": {"nodes": [ 0.1702798691054306,  0.9037017767993690,  2.2510866298658319,  4.2667001702876101, 7.0459054023932791, 10.7585160101801198, 15.7406786414777595, 22.8631317355938186],
        "weights": [3.6918858934277717e-01, 4.1878677737225132e-01, 1.7579498648834413e-01, 3.3343492244199105e-02, 2.7945362366839077e-03, 9.0765087708856083e-05, 8.4857467167664805e-07, 1.0480011681283307e-09]
    }
}

#check integration direction
def gauss_quadrature(f, method="legendre", a = 1, b = -1):
    key = method.lower()
        
    table = Tabs[key]
    nodes = table["nodes"]
    weights = table["weights"]
    
    sum = 0
    
    if key == "legendre":
        c1 = (b - a) / 2.0
        c2 = (b + a) / 2.0
        for x, w in zip(nodes, weights):
            sum = sum + w * f(c1 * x + c2)
        sum = sum * c1
        
    elif key == "laguerre":
        for x, w in zip(nodes, weights):
            sum = sum + w * f(x)
            
    return sum

def forward_euler(f, x0, y0, xend, h):
    steps = int(round((xend - x0) / h))
    x_vals, y_vals = [x0], [y0]
    
    x, y = x0, y0
    for i in range(steps):
        y = y + h * f(x, y)
        x = x + h
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals


def backward_euler(f, dfdy, x0, y0, xend, h):
    steps = int(round((xend - x0) / h))
    x_vals, y_vals = [x0], [y0]
    x, y = x0, y0
    
    for i in range(steps):
        xn = x + h
        yc = y

        def g(yn):
            return yn - yc - h * f(xn, yn)
            
        def dg(yn):
            return 1.0 - h * dfdy(xn, yn)
        yguess = yc + h * f(x, yc)
        
        nr_result = newton_raphson(g, dg, yguess)[0]
        if nr_result is not None:
            y = nr_result
        else:
            y = yguess 
            
        x = xn
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals


def predictor_corrector(f, x0, y0, xend, h):
    steps = int(round((xend - x0) / h))
    x_vals, y_vals = [x0], [y0]
    
    x, y = x0, y0
    for i in range(steps):
        fc = f(x, y)
        y_predict = y + h * fc
        xn = x + h
        y = y + (h / 2.0) * (fc + f(xn, y_predict))
        x = xn
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals

def rk4_step(f, x, y, h):
    """
    f: Function returning a list of derivatives given (x, y)
    x: Current independent variable
    y: Current state vector (list)
    h: Step size
    """
    # k1
    d1 = f(x, y)
    k1 = [h * d for d in d1]
    
    # k2
    y2 = [y[i] + 0.5 * k1[i] for i in range(len(y))]
    d2 = f(x + 0.5 * h, y2)
    k2 = [h * d for d in d2]
    
    # k3
    y3 = [y[i] + 0.5 * k2[i] for i in range(len(y))]
    d3 = f(x + 0.5 * h, y3)
    k3 = [h * d for d in d3]
    
    # k4
    y4 = [y[i] + k3[i] for i in range(len(y))]
    d4 = f(x + h, y4)
    k4 = [h * d for d in d4]
    
    # Update next state
    y_next = [y[i] + (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]) / 6.0 for i in range(len(y))]
    return y_next

def rk4_solve(f, x0, y0, x_end, h):
    x_list = [x0]
    y_list = [y0]
    
    x = x0
    y = y0
    
    steps = int(round((x_end - x0) / h))
    for _ in range(steps):
        y = rk4_step(f, x, y, h)
        x += h
        x_list.append(x)
        y_list.append(y)
        
    return x_list, y_list