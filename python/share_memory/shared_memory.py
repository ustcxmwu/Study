# from multiprocessing import Process, Value, RawArray
import multiprocessing as mp
import numpy as np


def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


def extract_params_as_shared_arrays(model):
    """
    converts params to shared arrays
    """
    # can get in the form of list -> shared + policy + value
    shared_arrays = []

    weights_dict = model.get_all_weights()
    weight_list = []

    for k,v in weights_dict.items():
        weight_list += v

    for weights in weight_list:
        shared_arrays.append(mp.RawArray('f', weights.ravel()))
    return shared_arrays


def mpraw_as_np(shape, dtype):
    """Construct a numpy array of the specified shape and dtype for which the
    underlying storage is a multiprocessing RawArray in shared memory.

    Parameters
    ----------
    shape : tuple
      Shape of numpy array
    dtype : data-type
      Data type of array

    Returns
    -------
    arr : ndarray
      Numpy array
    """

    sz = int(np.product(shape))
    csz = sz * np.dtype(dtype).itemsize
    raw = mp.RawArray('c', csz)
    return np.frombuffer(raw, dtype=dtype, count=sz).reshape(shape)


if __name__ == '__main__':
    num = mp.Value('d', 0.0)
    arr = mp.Array('i', range(10))

    p = mp.Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])
