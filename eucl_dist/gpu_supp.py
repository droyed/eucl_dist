import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule as SM
import skcuda.misc as misc
from eucl_dist.cuda_kernels import sq_sum_codetext, addvecs_codetext

# Global (to this file) constants and functions
addvecs_bcast_gpu = SM(addvecs_codetext).get_function("add_vectors_broadcast")

BSZ, GSZ = 1024, 10
block_blocksize_define_str = "#define BLOCK_SIZE "+ str(BSZ)
block_gridsize_define_str = "#define GRID_SIZE "+ str(GSZ)
define_str = "\n".join([block_blocksize_define_str, block_gridsize_define_str])
sq_sum_gpu = SM(define_str + sq_sum_codetext).get_function("sq_sum")

def addvecs(a, b, output='gpu'):
    """
    Add two 1D arrays for all pairs of elements resulting in a 2D array.

    Parameters
    ----------
    A : ndarray or GPUArray
    B : ndarray or GPUArray
    output : str, optional
        Selects the output datatype. It can be 'cpu' or 'gpu'.

    Returns
    -------
    out : ndarray or GPUArray
        Pairwise summation of elements from input 1D arrays. Thus, if first
        array has M elements and second one has N elements, we would have an
        output array of shape (M,N). The output class would be GPUArray or
        ndarray class, depending on the input argument 'output'. This decides
        whether the final output is to be kept on the GPU or brought back to
        the CPU host respectively.

    """

    if str(type(a)).find('gpuarray')!=-1:
        a_gpu = a
    elif str(type(b)).find('ndarray')!=-1:
        a_gpu = gpuarray.to_gpu(a)
    else:
        raise Exception("Input type invalid")

    if str(type(b)).find('gpuarray')!=-1:
        b_gpu = b
    elif str(type(b)).find('ndarray')!=-1:
        b_gpu = gpuarray.to_gpu(b)
    else:
        raise Exception("Input type invalid")

    M, N = a_gpu.shape[0], b_gpu.shape[0]
    out_gpu = gpuarray.empty((M,N),dtype=np.float32)
    BSZ = min(1024,N)
    GSZ = M
    num_iter = int(np.ceil(N/float(1024)))
    a_shp = np.int32([M,N,num_iter])
    addvecs_bcast_gpu(out_gpu, a_gpu, b_gpu, drv.In(a_shp), \
                                block=(BSZ,1,1), grid=(GSZ,1))

    if output=='gpu':
        return out_gpu
    elif output=='cpu':
        return out_gpu.get()
    else:
        raise Exception("Output type invalid")

def addvecs_gpu(a_gpu, b_gpu):
    """
    Add two 1D arrays on GPU for all pairs of elements resulting in a 2D array.

    Parameters
    ----------
    A : GPUArray
    B : GPUArray

    Returns
    -------
    out : GPUArray
        Pairwise summation of elements from input 1D arrays of GPUArray class.
        If first array has M elements and second one has N elements, we would
        have an output array of shape (M,N). Output would reside on the GPU side.

    """

    M, N = a_gpu.shape[0], b_gpu.shape[0]
    out_gpu = gpuarray.empty((M,N),dtype=np.float32)
    BSZ = min(1024,N)
    GSZ = M
    num_iter = int(np.ceil(N/float(1024)))
    a_shp = np.int32([M,N,num_iter])
    addvecs_bcast_gpu(out_gpu, a_gpu, b_gpu, drv.In(a_shp), \
                                block=(BSZ,1,1), grid=(GSZ,1))
    return out_gpu

def sq_sums(a_gpu, b_gpu, GSZ=GSZ):
    """
    Compute squared summations of rows from GPUArrays and then their pairwise summations.

    Parameters
    ----------
    A : GPUArray
    B : GPUArray
    GSZ : int, optional
        Grid size for CUDA kernel invocation

    Returns
    -------
    out : GPUArray
        Compute squared summations of each row for each of the inputs on GPU
        giving us two 1D arrays. Then, compute the pairwise summation of
        elements from them, leading to a 2D array.
        The output would still reside on the GPU device.

    """

    M,N,R = a_gpu.shape[0], b_gpu.shape[0], a_gpu.shape[1]

    if R>2048:
        raise Exception("Number of columns > 2048 not yet supported!")
    BSZ = 2**int(np.ceil(np.log(R)/np.log(2))-1)

    out_gpu1 = gpuarray.empty((M),dtype=np.float32)
    out_gpu2 = gpuarray.empty((N),dtype=np.float32)

    shp = np.int32([M,R,N])
    sq_sum_gpu(out_gpu1, out_gpu2, a_gpu, b_gpu, drv.In(shp), block=(BSZ,1,1), grid=(GSZ,1))
    out_gpu = addvecs_gpu(out_gpu1, out_gpu2)
    return out_gpu

def squared_sum(a,b,method):
    """
    Compute squared summations of rows and then their pairwise summations.

    Parameters
    ----------
    A : ndarray
    B : ndarray
    method : str
        This chooses the method for the computations.
        It can be 'add_togpu' or 'togpu_misc_add' or 'togpu_cuda_add'.

    Returns
    -------
    out : GPUArray
        Compute squared summations of each row for each of the ndarrays giving us
        two 1D arrays. Then, compute their pairwise summations to result in a 2D
        array.
        There are three workflows, thus three possible values for the
        corresponding argument that chooses one of those values for : 'method'.

        They are listed below:

        'add_togpu' : Compute squared sum of rows of the inputs and then perform
        broadcasted  element-wise summations, all on CPU. Then, transfer this
        array to GPU as the output.

        'togpu_misc_add' : Compute squared sum of rows of the inputs, giving us
        two `1D` arrays. Transfer these as two arrays onto GPU. Create a `zeros`
        array directly on GPU and in two steps add in the two summed arrays in a
        broadcasted manner, using 'skcuda.misc.add.add_matvec' along the rows and
        columns, giving us the pairwise summations.

        'togpu_cuda_add' : Same as previous one, but instead of using
        'skcuda.misc.add.add_matvec', we would roll out our own CUDA kernel,
        with the idea of having more control, specifically making use of
        threads and blocks and in the process attaining best possible performance.        

    """

    c_gpu = None # Initialize output

    if method=="add_togpu":
        c = np.einsum('ij,ij->i',a,a)[:,None] + np.einsum('ij,ij->i',b,b)
        c_gpu = gpuarray.to_gpu(c)

    elif method=="togpu_misc_add":
        a1_gpu = gpuarray.to_gpu(np.einsum('ij,ij->i',a,a)[:,None])
        b1_gpu = gpuarray.to_gpu(np.einsum('ij,ij->i',b,b))

        M,N = a.shape[0], b.shape[0]
        c_gpu = gpuarray.zeros((M,N),dtype=np.float32)
        misc.add_matvec(c_gpu, a1_gpu, out=c_gpu)
        misc.add_matvec(c_gpu, b1_gpu, out=c_gpu)

    elif method=="togpu_cuda_add":
        c_gpu = addvecs(np.einsum('ij,ij->i',a,a), np.einsum('ij,ij->i',b,b))

    else:
        raise Exception("Invalid method.")

    return c_gpu

def convert_f32(a):
    """
    Convert to float32 dtype.

    Parameters
    ----------
    a : ndarray

    Returns
    -------
    out : ndarray
        Converts to float32 dtype if not already so. This is needed for
        implementations that work exclusively work such datatype.

    """

    if a.dtype!=np.float32:
        return a.astype(np.float32)
    else:
        return a
