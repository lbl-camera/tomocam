import numpy as np
import tomocam

a = np.arange(20, dtype=np.float32)
A = tomocam.DistArray(a)
print('A = ')
print(A.to_numpy())
b = np.ones(20, dtype=np.float32)
B = tomocam.DistArray(b)
print('B = ')
print(B.to_numpy())

print('subtrac A from B')
A -= B
print(A.to_numpy())
print('Shape of A')
print(A.shape)
d = A.to_numpy()
print(d.shape)
