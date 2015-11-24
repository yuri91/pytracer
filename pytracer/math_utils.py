import numpy as np

def normalize(x):
    x /= np.linalg.norm(x)
    return x

def cross(x,y):
    i = x[1]*y[2]-x[2]*y[1]
    j = x[2]*y[0]-x[0]*y[2]
    k = x[0]*y[1]-x[1]*y[0]
    
    v = np.array([i,j,k])

    return v
