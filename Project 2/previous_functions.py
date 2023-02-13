# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt



# Function to calculate the infinity norm sometimes called max norm
def max_norm(M):
    
    # Get the absolute value of the matrix
    M_abs = np.absolute(M)
    
    # Sum each row of the absolute matrix
    M_sum = np.sum(M_abs, axis=1)
    
    # Return the largest row sum
    return np.max(M_sum) 



# We define a function to do calculate the condition number of a matrix
def cond(M):
    
    # Calculate the inverse matrix M^-1
    Minv = np.linalg.inv(M)
    
    # Calculate the max norms and multiply them together to get the condition number
    cond_number = max_norm(M) * max_norm(Minv)
    
    return cond_number



# Define a function to perform LU factorization on a square matrix M
def lu_factorize(M):
    
    # Check that the input matrix is in fact square
    if M.shape[0] != M.shape[1]:
        print('Input matrix is not square')
    
    # Initialise L as an identity matrix of the same dimensions as M
    n = M.shape[0]
    L = np.identity(n)
    A = M.copy() #copy to not change input
    U = np.zeros(A.shape)
    
    # Loop over all columns except the last one
    for k in range(n-1):
        
        # Stop if the pivot is zero (the element in the diagonal of given column)
        if A[k,k] == 0:
            print('Encountered pivot equal to zero')
            return
        
        # Loop over the subdiagonal rows in the given column
        for i in range(k+1, n):
            
            # Find the value for all subdiagonal elements in L
            L[i,k] = A[i,k] / A[k,k]
            
        # Loop again, combinations of j,i,k will give us elements of the submatrix in bottom right    
        for j in range(k+1,n):
            for i in range(k+1,n):
                
                # Apply transoformation to the remaining submatrix
                A[i,j] = A[i,j] - L[i,k]*A[k,j]
    
        # Assign upper triangular part of the transformed M to U
        U[:k+1,k] = A[:k+1,k] #rows, columns k+1 because the last element is not included when using :
    
    # Assign also the last column which is not used above because we range until n-1
    U[:, -1] = A[:, -1]
    
    return L, U



# We define the function
def forward_substitute(L, b):
    
    # Make empty array for solution vector y
    n = L.shape[0]
    y_vec = np.zeros(n)
    
    # Make copy of b so we dont change the values when updating
    b_copy = b.copy()
    
    # Loop over columns in the L matrix
    for j in range(n):
        
        # Stop if matrix is singular
        if L[j,j] == 0:
            print('Singular matrix encountered. Cannot procede.')
            return
        
        # Compute solution component, i.e. the solution to the i'th linear equation
        y_vec[j] = b_copy[j] / L[j,j]
        
        # Update right hand side
        for i in range(j+1,n):
            b_copy[i] = b_copy[i] - L[i,j] * y_vec[j]
        
    return y_vec



def back_substitute(U,y):
    
    # Make empty array for solution vector x
    n_rows, n_cols = U.shape
    x_vec = np.zeros(n_cols)
    
    # Make copy of y so we dont change the values when updating
    y_copy = y.copy()
    
    # Loop backwards over columns
    
    for j in reversed(range(n_rows)):
        
        # Stop if matrix is singular
        if j+1 > n_cols: continue
        else:
            if U[j,j] == 0:
                print('Singular matrix encountered. Cannot procede.')
                return
        
        # Compute solution component
        x_vec[j] = y_copy[j] / U[j,j]
        
        for i in range(0,j):
            y_copy[i] = y_copy[i] - U[i,j] * x_vec[j]
            
    return x_vec



# Make a function that combines it
def linear_solver(A,b): 
    """
    A = coefficient matrix
    b = dependent variable values
    """
    A = A.copy()
    L, U = lu_factorize(A)
    y = forward_substitute(L, b)
    x = back_substitute(U, y)
    
    return x



def householder_qr(A, rounded=False, decimals=4):
    """
    Description
    """
    
    # Inititalise R as a copy of A to not change the original matrix - same goes for b
    R = A.copy()
    
    # Get dimensions of matrix A
    n_rows, n_cols = A.shape
    
    # Check that m>n so the procedure is possible
    if n_rows<n_cols:
        print('Encountered matrix with more columns than rows. System is unsolvable.')
        return
    
    # Initialise Q as an identity matrix
    Q = np.identity(n_rows)
    
    ######################################
    ### Compute the householder vector ###
    ######################################
    
    # Loop over columns
    for j in range(n_cols):
        
        # Let a be the j'th column of the matrix A or submatrix depending on iteration
        a = R[j:,j]
        
        # Calculate the norm of a
        norm_a = np.sqrt( np.sum(a**2) )
        
        # Calculate the alpha which is the proper sign (to avoid nummerical cancellation errors) times the norm
        alpha = -np.sign(a[0]) * norm_a
        
        # Make the first standard basis vector for the matrix or subbmatrix
        e = np.zeros_like(a)
        e[0] = 1
        
        # Initialise a householder vector 'u': the vector 'a' minus 'alpha' on the diagonal element
        v = a - alpha*e
        
        # Normalise it
        v = v/np.sqrt(np.sum(v**2))

        """
        check that something is not zero - is it a[j]?
        """
        #############################################
        ### Compute the householder vector matrix ###
        #############################################
    
        # Compute the Householder matrix (notice that H_n with n>0 are embedded into a mxn identity)
        H = np.identity(n_rows)
        H[j:,j:] -= (2 / np.dot(v, v)) * np.dot(v[:, None], v[None, :]) #normalise twice?

        #################################
        ### Compute Q,R and b' from H ###
        #################################
        
        # Q = I*H_1*H_2...
        Q = Q@H
        
        # Get R from H_3*H_2*H_1*A=R_3
        R = H@R
        
    if rounded:
        Q = np.round_(Q, decimals=decimals)
        R = np.round_(R, decimals=decimals)
            
    return Q, R



def least_squares(A,b):
    
    # Perform QR factorisation on A
    Q, R = householder_qr(A)
    
    # Get the transformed right hand side vector b
    b_trans = Q.T@b
    
    # Solve the upper rectangular matrix R with back substitution
    n_rows, n_cols = A.shape
    x = back_substitute(R,b_trans) #why do we need to cut it off here?
    
    return x




