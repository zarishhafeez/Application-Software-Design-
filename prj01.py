import numpy as np
import easynn as nn

# Create a numpy array of 10 rows and 5 columns.
# Set the element at row i and column j to be i+j.
def Q1():
    nump  = np.arange(50).reshape(10,5)
    for i in range (np.shape(nump)[0]):
        for j in range (np.shape(nump)[1]):
            nump[i][j]=i+j
    return nump

# Add two numpy arrays together.
def Q2(a, b):
    return a+b

# Multiply two 2D numpy arrays using matrix multiplication.
def Q3(a, b):
    ##res = [[0 for x in range]]
    multi = np.matmul(a,b)
    return multi

# For each row of a 2D numpy array, find the column index
# with the maximum element. Return all these column indices.
def Q4(a):
    arrays = []
    for i in range (np.shape(a)[0]):
        arrays.append(list(a[i]).index(max(a[i])))
    return np.array(arrays)

# Solve Ax = b.
def Q5(A, b):
    x = np.linalg.solve(A,b)
    return x

# Return an EasyNN expression for a+b.
def Q6():
    a = nn.Input("a")
    b = nn.Input("b")
    c = a+b
    return c

# Return an EasyNN expression for a+b*c.
def Q7():
    a = nn.Input("a")
    b = nn.Input("b")
    c = nn.Input("c")
    e = a + b*c
    return e

# Given A and b, return an EasyNN expression for Ax+b.
def Q8(A, b):
    A = nn.Const(A)
    b = nn.Const(b)
    x = nn.Input("x")
    f = A*x +b
    return f

# Given n, return an EasyNN expression for x**n.
def Q9(n):
    temp = 1
    x = nn.Input("x")
    for _ in range (n):
        temp = x *temp
    return temp

# Return an EasyNN expression to compute
# the element-wise absolute value |x|.
def Q10():
    a = nn.ReLU()
    x = nn.Input("x")
    b = a(x)
    c = a(-x)
    return b+c
