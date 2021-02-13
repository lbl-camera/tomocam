

from .darray import DistArray
from .transform import radon, iradon
from .modeling import calc_gradients, update_total_variation, MBIR

def axpy(alpha, y, x):
    cTomocam.axpy(alpha, y.handle, x.handle)

def norm(x):
    return cTomocam.norm(x)
