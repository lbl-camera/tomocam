import numpy as np
import arrayfire as af

def padmat(X, size, value=0, dtype=np.float32):
    """
    pads X to size with constant values

    Parameters:
    -----------
    X: np.ndarray
        2-D Numpy array representing an image
    size: tuple
        New size of the padded matrix
    value: scalar, default = 0
        Fill value
    dtype: np.dtype, default = float32
        data-type of the output image

    Returns:
    --------
    np.ndarray,
        padded image
    """
    if len(size) == 1:
        size = (size[0], size[0])
    if size[0] < X.shape[0] or size[1] < X.shape[1]:
        raise ValueError('Dims of padded image are smaller than the input')
    Y = np.full(size, value, dtype=dtype)
    i = (size[0] - X.shape[0])//2
    j = (size[1] - X.shape[1])//2
    Y[i:i+X.shape[0],j:j+X.shape[1]] = X
    return Y


def np2af(arr):
    """
    Cast numpy to arrayfire while preserving data order

    Parameters:
    -----------
    arr: np.ndarray
        n-dimensional numpy array

    Returns:
    --------
    af.Array
        arrayfire array
 
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError('Input is not a numpy array')
    dims = range(arr.ndim)[::-1]
    return af.reorder(af.np_to_af_array(arr), *dims)

def af2np(af_arr):
    """
    Cast arrayfire to numpy array while preserving data order

    Parameters:
    -----------
    arr: af.Array
        n-dimensional arrayfire array

    Returns:
    --------
    np.ndarray
        numpy array
 
    """

    if not isinstance(af_arr, af.Array):
        raise TypeError('Input is not a arrayfire array')

    dims = range(af_arr.numdims())[::-1]
    return af_arr.__array__().transpose(*dims)


@af.broadcast
def multiply(arr, vec):
    """
    multiply rhs to a multi-dimensional array row-wise or slice-wise

    Parameters:
    -----------
    arr: af.Array,
        Multi-dimensional arrayfire array
    vec: af.Array,
        1-d or 2-d arrayfire array 
    
    Returns:
    --------
    af.Array,
        result of multiplication
    """

    if vec.numdims() == 1 and arr.shape[0] != vec.shape[0]:
        raise ValueError('Shape mismatch')
    elif vec.numdims() == 2 and arr.shape[:2] != vec.shape:
        raise ValueError('Shape mismatch')
    else:
        raise ValueError('Unsupported rhs shape')
    return arr * vec
