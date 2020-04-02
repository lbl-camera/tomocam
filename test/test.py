import numpy as np
import tomocam

a = np.arange(20, dtype=np.float32)
A = tomocam.DistArray(a)
print(A.to_numpy())
b = np.ones(20, dtype=np.float32)
B = tomocam.DistArray(b)
print(B.to_numpy())

A -= B
print(A.to_numpy())
print(A.shape)
d = A.to_numpy()
print(d.shape)
