from . import cTomocam


def set_num_of_gpus(num):
    """Sets the number of GPUs tomocam is allowed to use.

    By default tomocam uses every visible GPU. Calls made after this function
    will be distributed over at most `num` devices.

    Parameters
    ----------
    num: int (> 0)
        Maximum number of GPUs to use
    """
    num = int(num)
    if num < 1:
        raise ValueError('number of GPUs must be greater than 0')
    cTomocam.set_num_of_gpus(num)
