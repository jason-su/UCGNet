import numpy as np

a = np.array([[1,1,0,0],[0,0,1,0],[1,1,0,1],[1,0,1,0]])

print(a)
b = np.dot(a,a)
print(b)
d = np.dot(b,a)
print(np.sum(d))
c = np.dot(b,b)
print(c)
