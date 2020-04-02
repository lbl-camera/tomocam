import numpy as np
from .cTomocam import axpy
from .cTomocam import norm
from .cTomocam import DArray

class DistArray:
    def __init__(self, array):
        if not isinstance(array, np.ndarray):
            raise TypeError('argument to DistArray must be a numpy.ndarray')
        self.handle = DArray(array)
        self.shape = self.handle.shape()

    def norm(self):
        return norm(self.handle)

    def to_numpy(self):
        return self.handle.to_numpy()

    def __add__(self, other):
        if not isinstance(other, DistArray):
            raise TypeError('operand type mismatch')
        if self.shape != other.shape:
            raise ValueError('dimension mismatch')
        axpy(1, other.handle, self.handle)
        return self

    def __iadd__(self, other):
        if not isinstance(other, DistArray):
            raise TypeError('operand type mismatch')
        if self.shape != other.shape:
            raise ValueError('dimension mismatch')
        axpy(1, other.handle, self.handle)
        return self

    def __sub__(self, other):
        if not isinstance(other, DistArray):
            raise TypeError('operand type mismatch')
        if self.shape != other.shape:
            raise ValueError('dimension mismatch')
        axpy(-1, other.handle, self.handle)
        return self

    def __isub__(self, other):
        if not isinstance(other, DistArray):
            raise TypeError('operand type mismatch')
        if self.shape != other.shape:
            raise ValueError('dimension mismatch')
        axpy(-1, other.handle, self.handle)
        return self

