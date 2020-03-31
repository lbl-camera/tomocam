import numpy as np
from .cTomocam import axpy
from .cTomocam import norm
from .cTomocam import DArray

class DistArray(DArray):
    def __int__(self, array):
        super().__init__(array)

    def norm(self):
        return norm(self)

    def __add__(self, d_array):
        if not isinstance(d_array, DArray):
            raise TypeError('operand type mismatch')
        if self.dims != d_array.dims:
            raise ValueError('dimension mismatch')
        axpy(1, self, d_array)
        return self

    def __iadd__(self, d_array):
        if not isinstance(d_array, DArray):
            raise TypeError('operand type mismatch')
        if self.dims != d_array.dims:
            raise ValueError('dimension mismatch')
        axpy(1, self, d_array)

    def __sub__(self, d_array):
        if not isinstance(d_array, DArray):
            raise TypeError('operand type mismatch')
        if self.dims != d_array.dims:
            raise ValueError('dimension mismatch')
        axpy(-1, self, d_array)
        return self

    def __isub__(self, d_array):
        if not isinstance(d_array, DArray):
            raise TypeError('operand type mismatch')
        if self.dims != d_array.dims:
            raise ValueError('dimension mismatch')
        axpy(-1, self, d_array)
