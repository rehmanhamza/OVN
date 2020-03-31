import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Exercise SET 3

'''
#1
#a = np.array([[1,2], [3,4], [5,6], [7,8]])
a = np.empty([4,2], dtype = np.uint16)
print(a)
print(a.shape)
print(a.ndim)

# 2
a = np.arange(100, 200, 10).reshape(5,2)
print(a)


# 3
a = np.array([[11 ,22, 33], [44, 55, 66], [77, 88, 99]])
b = a[..., 1]
print(b)


# 4
a = np.array([[3 ,6, 9, 12], [15 ,18, 21, 24], [27 ,30, 33, 36], [39 ,42, 45, 48], [51 ,54, 57, 60]])
a = a[::2, 1::2]
print(a)


# 5
a = np.array([[5, 6, 9], [21 ,18, 27]])
b = np.array([[15 ,33, 24], [4 ,7, 1]])
c = np.add(a,b)
c = np.square(c)

print(c)


# 6
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
a = np.sort(a)

print(a)


# 7
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
a1 = np.amin(a, 1)
a2 = np.amax(a, 0)
print(a1, "\n", a2)


# 8
a = np.array([[34,43,73],[82,22,12],[53,94,66]])
new_column = np.array([10,10,10])
a = np.delete(a, 1, axis=1)
a = np.insert(a, 1, new_column, axis=1)

print(a)
'''

# Exercise SET 4
