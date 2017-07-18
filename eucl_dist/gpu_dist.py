import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as culinalg
import skcuda.misc as misc
from eucl_dist.gpu_supp import sq_sums, squared_sum, convert_f32

culinalg.init()
misc.init()

def dot_accum(a,b):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using GPU. This computes the matrix-multiplication of
    the GPU versions of the inputs, gets it back to host CPU and then
    accumulates the squared sum of rows into it along the rows and columns
    respectively.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.

    Returns
    -------
    out : ndarray
        This holds the euclidean distances.

    """

    a_gpu = gpuarray.to_gpu(-2*a)
    b_gpu = gpuarray.to_gpu(b)
    out = culinalg.dot(a_gpu, b_gpu, transb='T').get()
    out += np.einsum('ij,ij->i',a,a)[:,None]
    out += np.einsum('ij,ij->i',b,b)
    return out

def sqsum_adddot(a,b,method):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using GPU. This uses the input arrays themselves to
    compute element-wise summations of squared sum of rows and accumulates into
    the matrix-multiplication result residing on GPU.
    The final result resides on GPU.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    method : str
        It can be 'add_togpu' or 'togpu_misc_add' or 'togpu_cuda_add'.
        Refer to function "squared_sum" for more information.

    Returns
    -------
    out : GPUArray
        This holds the euclidean distances residing on GPU.
    """

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = squared_sum(a,b,method=method)
    return culinalg.add_dot(a_gpu, b_gpu, c_gpu,  transb='T', alpha=-2.0)

def sqsum_adddot2(a,b):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using GPU. This uses the GPUArray versions of the
    input arrays to compute element-wise summations of squared sum of rows and
    accumulates into the matrix-multiplication result residing on GPU.
    The final result resides on GPU.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.

    Returns
    -------
    out : GPUArray
        This holds the euclidean distances.

    """

    a_gpu = gpuarray.to_gpu(a)
    b_gpu = gpuarray.to_gpu(b)
    c_gpu = sq_sums(a_gpu, b_gpu)
    return culinalg.add_dot(a_gpu, b_gpu, c_gpu,  transb='T', alpha=-2.0)

def dist(a, b, optimize_level=0, output="cpu"):
    """
    Compute squared euclidean distance between two 2D arrays representing
    n-dimensional points using GPU.

    Parameters
    ----------
    A : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    B : ndarray
        2D NumPy array of float dtype representing n-dimensional points, with
        each row being one point.
    optimize_level : int, optional
        Selects one of the five methods to compute euclidean distances.
        This can be any number between 0 to 4, with their significance being
        listed below -
        0:  Transfer input arrays to GPU.
            Perform matrix-multiplication of them on GPU.
            Transfer them back to CPU.
            Into it, add in squared summation of rows from the two
            input arrays in two separate steps allowing for broadcasting.

        1:  Perform squared summation of rows on CPU.
            Do their element-wise additon on CPU to obtain a 2D array.
            Transfer it to GPU.
            Add into it matrix multiplication of inputs arrays computed with GPU.

        2:  Perform squared summation of rows on CPU.
            Transfer these to GPU.
            Do their element-wise additon on GPU using skcuda module.
            Add it to the matrix multiplication of input arrays obtained on
            GPU using CUBLAS again supported by skcuda module.

        3:  Perform squared summation of rows on CPU.
            Transfer them to GPU.
            Do their element-wise additon on GPU using custom CUDA kernel.
            Add it to the matrix multiplication of input arrays obtained on GPU.

        4:  Transfer input arrays to GPU.
            Perform squared summation of their rows on GPU using custom CUDA kernel.
            Add it to the matrix multiplication of input arrays obtained on GPU.

    output : str, optional
        Selects whether to keep the final data on CPU or GPU.
        With optimize_level = 1 till 4, we have the final result on GPU.
        So, with those four options we could retrieve it back to host CPU or
        keep it on GPU if we intend to do further operations on it.
        This can be 'cpu' or 'gpu'.

    Returns
    -------
    out : ndarray or GPUArray
        It holds the euclidean distances. This would be NumPy ndarray or PyCUDA
        GPUArray based on the argument 'output'.

    Example(s)
    -------
    Find the pairwise euclidean distances between three 2-D coordinates:

    >>> from from eucl_dist.gpu_dist import dist
    >>> coords = np.array([[2,3],[3,4],[2,5]])
    >>> dist(coords, coords)
    array([[ 0.,  2.,  4.],
           [ 2.,  0.,  2.],
           [ 4.,  2.,  0.]], dtype=float32)

    """

    A = convert_f32(a)
    B = convert_f32(b)

    if optimize_level==0:
        out = dot_accum(A,B)
    elif optimize_level==1:
        out = sqsum_adddot(A,B,method='add_togpu')
    elif optimize_level==2:
        out = sqsum_adddot(A,B,method='togpu_misc_add')
    elif optimize_level==3:
        out = sqsum_adddot(A,B,method='togpu_cuda_add')
    elif optimize_level==4:
        out = sqsum_adddot2(A,B)
    else:
        raise Exception("Invalid entry for optimize_level")

    if output=="cpu":
        if optimize_level==0:
            return out
        else:
            return out.get()
    elif output=="gpu":
        if optimize_level==0:
            raise Exception("Optimize level - 0 not supported with GPU output")
        else:
            return out
    else:
        raise Exception("Invalid entry for output")
