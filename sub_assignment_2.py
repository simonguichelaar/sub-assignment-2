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
    
    # Calculate P (Peclet number): P = v*a/D
    P = v * a / D
    
    # Initialize matrix A and vector f for n internal nodes
    # The scaled equation is: (1/P) * d²c/dx² - dc/dx - rc = 0
    # Using central differences:
    # (1/P) * (c_{i-1} - 2*c_i + c_{i+1})/h² - (c_{i+1} - c_{i-1})/(2h) - r*c_i = 0
    
    A = np.zeros((n, n))
    f = np.zeros(n)
    
    # Assemble the finite difference system
    for i in range(n):
        # Coefficients for the discretized equation
        # Multiply through by h² to avoid small coefficients
        # (1/P) * (c_{i-1} - 2*c_i + c_{i+1}) - (h/2P) * (c_{i+1} - c_{i-1}) - r*h²*c_i = 0
        
        coeff_left = 1.0/P - (h/(2*P))  # coefficient of c_{i-1}
        coeff_center = -2.0/P - r*h**2  # coefficient of c_i
        coeff_right = 1.0/P + (h/(2*P))  # coefficient of c_{i+1}
        
        # Diagonal coefficient
        A[i, i] = coeff_center
        
        # Right-hand side (zero for interior)
        f[i] = 0.0
        
        # Left neighbor (i-1)
        if i > 0:
            A[i, i-1] = coeff_left
        else:
            # First interior node: boundary condition c(0) - (h/P)*dc/dx(0) = 1
            # This gives: c_0 - (h/P)*(c_1 - c_0)/(2h) = 1
            # So: c_0 - (1/(2P))*(c_1 - c_0) = 1
            # Rearranging: (1 + 1/(2P))*c_0 - (1/(2P))*c_1 = 1
            # We substitute c_0 = 1 + (1/(2P))*c_1 into the first equation
            # After substitution, the contribution is: coeff_left * 1 (boundary value)
            f[i] += coeff_left * 1.0  # c_0 = 1 boundary condition
        
        # Right neighbor (i+1)
        if i < n - 1:
            A[i, i+1] = coeff_right
        else:
            # Last interior node: dc/dx(L) = 0
            # Central difference: (c_{n+1} - c_{n-1})/(2h) = 0
            # So: c_{n+1} = c_{n-1}
            # This affects the last equation as: coeff_right * c_{n-1}
            A[i, i-1] += coeff_right  # Add the c_{n+1} = c_{n-1} contribution
    
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
    # Boundary conditions: c(0) = 1, dc/dx(L) = 0
    w_num = np.zeros(n + 2)
    w_num[0] = 1.0  # boundary condition at x=0: c(0) = 1
    w_num[1:-1] = c_internal  # internal nodes
    # w_num[-1] is determined by dc/dx(L) = 0 condition (handled in assembly)
    # For dc/dx(L) = 0: c_{n+1} = c_{n-1}, so we set w_num[-1] = w_num[-2]
    w_num[-1] = w_num[-2]  # boundary condition at x=L: dc/dx = 0
    
    ''' end of lines to be changed '''
    
    w_num = w_num.flatten()
    x_num = x_num.flatten()
    
    return x_num, w_num

# Set the information needed to perform the numerical simulation

# define the number of internal grid nodes which should be used in the numerical simulation
n = 9

# define the value of the parameter v (advection velocity)
v = 5e-3

# Define parameters based on the problem
a = 4  # average grain diameter
L = 55  # domain length
D = 0.002  # diffusion coefficient
k = 6.0 * 10**-5  # reaction rate
r = k * a / v  # scaled reaction coefficient

# Run the simulation
x_num, w_num = FiniteDifferenceMethod(n, a, L, D, k, v)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_num, w_num, 'b-o', linewidth=2, markersize=6)
plt.xlabel('x')
plt.ylabel('c(x)')
plt.title(f'Numerical Solution (n={n}, v={v}, L={L}, D={D}, k={k})')
plt.grid(True, alpha=0.3)
plt.show()

# Print results
print(f"Grid nodes: {x_num}")
print(f"Solution values: {w_num}")
print(f"Peclet number P = v*a/D = {v*a/D}")
