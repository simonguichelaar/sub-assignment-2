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
    x_num = np.linspace(0, L, n + 2)
    h = L / (n + 1)  # grid spacing
    
    # Initialize matrix A and vector f for n internal nodes
    A = np.zeros((n, n))
    f = np.zeros(n)
    
    # Assemble the finite difference system
    # For interior nodes: -(D/h²)*w_{i-1} + (k + 2*D/h²)*w_i - (D/h²)*w_{i+1} = v
    # For the first internal node (i=0): boundary condition at x=0 affects the equation
    # For the last internal node (i=n-1): boundary condition at x=L affects the equation
    
    for i in range(n):
        # Diagonal coefficient
        A[i, i] = k + 2 * D / h**2
        
        # Right-hand side
        f[i] = v
        
        # Left neighbor (i-1)
        if i > 0:
            A[i, i-1] = -D / h**2
        else:
            # First interior node: boundary condition w_0 = 0 contributes to RHS
            f[0] += 0  # w_0 = 0, so no contribution
        
        # Right neighbor (i+1)
        if i < n - 1:
            A[i, i+1] = -D / h**2
        else:
            # Last interior node: boundary condition w_{n+1} = 0 contributes to RHS
            f[n-1] += 0  # w_{n+1} = 0, so no contribution
    
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

    # Solve the linear system A * w_internal = f
    w_internal = spsl.spsolve(A, f)
    
    # add Dirichlet boundary conditions to w_num (if applicable)
    # Boundary conditions: w(0) = 0, w(L) = 0
    w_num = np.zeros(n + 2)
    w_num[0] = 0  # boundary condition at x=0
    w_num[1:-1] = w_internal  # internal nodes
    w_num[-1] = 0  # boundary condition at x=L
    
    ''' end of lines to be changed '''
    
    w_num = w_num.flatten()
    x_num = x_num.flatten()
    
    return x_num, w_num

# Set the information needed to perform the numerical simulation

# define the number of internal grid nodes which should be used in the numerical simulation
n = 9

# define the value of the parameter v
v = 5e-3

# Define additional parameters (you need to set these based on your problem)
# These are typical values - adjust based on your specific problem
a = 1.0    # parameter a
L = 1.0    # domain length
D = 1.0    # diffusion coefficient
k = 0.0    # reaction coefficient (if applicable)

# Run the simulation
x_num, w_num = FiniteDifferenceMethod(n, a, L, D, k, v)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_num, w_num, 'b-o', linewidth=2, markersize=6)
plt.xlabel('x')
plt.ylabel('w(x)')
plt.title(f'Numerical Solution (n={n}, v={v})')
plt.grid(True, alpha=0.3)
plt.show()

# Print results
print(f"Grid nodes: {x_num}")
print(f"Solution values: {w_num}")
