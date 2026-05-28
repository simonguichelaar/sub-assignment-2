# Code for Sub-Assignment 2

# import extra modules/functions
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

# define the function that builds the matrix A and the vector f
def Assemble_Matrix_and_Vector(n,a,L,D,k,v):
    # INPUT
    # n : the number of internal grid nodes
    # a: The value of parameter a to use within this function.
    # L: The value of parameter L to use within this function.
    # D: The value of parameter D to use within this function.
    # k: The value of parameter k to use within this function.
    # v: The value of parameter v to use within this function.
    
        
    # OUTPUT
    # x_num : a NUMPY array containing ALL grid nodes
    # A : the matrix of the finite difference scheme., this should be a SCIPY matrix with format CSC (which is done for you)
    # f : the vector of the finite difference scheme, this should be a NUMPY array
    
    ''' start of lines to be changed '''
    # Create grid with n+2 total nodes (n internal + 2 boundary)
    # Use linspace as instructed (not arange)
    x_num = np.linspace(0, L, n + 2)
    h = L / (n + 1)  # grid spacing
    
    # Calculate scaled parameters
    P = v * a / D  # Peclet number
    r = k * a / v  # scaled reaction coefficient
    
    # Initialize matrix A and vector f for n internal nodes
    A = np.zeros((n, n))
    f = np.zeros(n)
    
    # Assemble the finite difference system
    # Scaled equation: (1/P) * d²c/dx² - dc/dx - rc = 0
    # Using central differences
    
    for i in range(n):
        # Coefficients for the discretized equation at node i
        coeff_left = 1.0/P - h/(2*P)    # coefficient of c_{i-1}
        coeff_center = -2.0/P - r*h**2  # coefficient of c_i
        coeff_right = 1.0/P + h/(2*P)   # coefficient of c_{i+1}
        
        # Diagonal coefficient
        A[i, i] = coeff_center
        
        # Right-hand side
        f[i] = 0.0
        
        # Left neighbor (i-1)
        if i > 0:
            A[i, i-1] = coeff_left
        else:
            # First interior node: boundary condition c(0) - (h/P)*dc/dx(0) = 1
            # This contributes to RHS: c_0 = 1
            f[i] += coeff_left * 1.0
        
        # Right neighbor (i+1)
        if i < n - 1:
            A[i, i+1] = coeff_right
        else:
            # Last interior node: dc/dx(L) = 0 means c_{n+1} = c_{n-1}
            A[i, i-1] += coeff_right
    
    ''' end of lines to be changed '''
       
    # convert A and f to a better format
    if not (A is None):
        if not (sps.issparse(A)):
            A = sps.csc_matrix(A)
        else:
            A = A.tocsc()

    f = f.flatten()
    x_num = x_num.flatten()
    
    return x_num, A, f

# Define the complete finite difference method
def FiniteDifferenceMethod(n,a,L,D,k,v):
    # INPUT
    # n : the number of internal grid nodes
    # a: The value of parameter a to use within this function.
    # L: The value of parameter L to use within this function.
    # D: The value of parameter D to use within this function.
    # k: The value of parameter k to use within this function.
    # v: The value of parameter v to use within this function.
    
    # OUTPUT
    # x_num : a NUMPY array containing all grid nodes
    # w_num : the numerical solution in all grid nodes in a NUMpY array of the same size as x_num
    
    # Assemble the large matrix and vector
    x_num,A,f = Assemble_Matrix_and_Vector(n,a,L,D,k,v)
    
    ''' start of lines to be changed '''
    
    # find the numerical solution (a linear system can be solved efficiently using the command spsl.spsolve,
    # documentation: https://scipy.github.io/devdocs/reference/generated/scipy.sparse.linalg.spsolve.html)

    # Solve the linear system A * c_internal = f
    c_internal = spsl.spsolve(A, f)
    
    # add Dirichlet boundary conditions to w_num (if applicable)
    # Boundary conditions: c(0) = 1 and dc/dx(L) = 0 (so c_{n+1} = c_{n-1})
    w_num = np.zeros(n + 2)
    w_num[0] = 1.0  # boundary condition at x=0
    w_num[1:-1] = c_internal  # internal nodes
    w_num[-1] = w_num[-2]  # boundary condition at x=L (from dc/dx = 0)
    
    ''' end of lines to be changed '''
    
    w_num = w_num.flatten()
    x_num = x_num.flatten()
    
    return x_num, w_num

# Set the information needed to perform the numerical simulation

# define the number of internal grid nodes which should be used in the numerical simulation
n = 9

# define the value of the parameter v
v = 5e-3

# Define additional parameters
a = 4  # average grain diameter
L = 55  # domain length
D = 0.002  # diffusion coefficient
k = 6.0 * 10**-5  # reaction rate

# Run the simulation
x_num, w_num = FiniteDifferenceMethod(n, a, L, D, k, v)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_num, w_num, 'b-o', linewidth=2, markersize=6)
plt.xlabel('x')
plt.ylabel('c(x)')
plt.title(f'Numerical Solution (n={n}, v={v})')
plt.grid(True, alpha=0.3)
plt.show()

# Print results
print(f"Grid nodes: {x_num}")
print(f"Solution values: {w_num}")
