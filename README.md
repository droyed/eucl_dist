# Euclidean Distance Computation in Python

## Introduction

Distance computations between datasets have many forms as listed in the [`Wiki page`](https://en.wikipedia.org/wiki/Distance). Among those, [`euclidean distance`](https://en.wikipedia.org/wiki/Euclidean_distance) is widely used across many domains. Computing it at different computing platforms and levels of computing languages warrants different approaches. At Python level, the most popular one is [`SciPy's cdist`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist). In the recent years, we have seen contributions from [`scikit-learn`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html) to the same cause. The motivation with this repository codebase is to bring in noticeable speedups by using an altogether different approach to distance computations and in the process leveraging parallel architectures like GPU. We will also study the best use cases that are suited to each of those implementations.


## Overview

There are two parts to the codebase - CPU and GPU based codes that as the names suggest, broadly speaking process the data on CPU and GPU respectively. The proposed method uses matrix-multiplication for better performance. With CPU implementations, we have two options - OpenBLAS or MKL to perform those matrix-multiplications. With GPU implementations, mainly we are leveraging CUBLAS to perform those matrix-multiplications. With most of the GPU implemenations presented in the codebase, we have the option to keep the final output on GPU to handle further compute heavy operations. This give us two options with GPU as well - keeping data on GPU or bringing it back to CPU host. Thus, considering both CPU and GPU implementations, there are four possibilities by which euclidean distances could be computed. There are few factors at play depending on the input data and output requirements that lets us propose different configurations for each of those four ways.

## Speedup figures

Between `SciPy's` `cdist` and `scikit-learn's` `pairwise_distances`, the runtime seems comparable, but `pairwise_distances` seemed to have an upper-hand in some cases. The speedups with the proposed methods over `pairwise_distances` using the best configurations for various dataset sizes thus obtained are listed below -

                   CPU-version:noMKL  CPU-version:MKL GPU-version:Out_CPU GPU-version:Out_GPU
    Dataset Size                                                                            
    5000  x    3                 2.3              4.2                 2.4                11.9
    10000 x    3                 2.1              3.8                 2.4                13.3
    5000  x   20                 6.8             11.9                 7.5                35.2
    500   x  300                28.3             53.1                29.2                33.8
    1000  x  300                33.8             56.7                57.6                83.2
    500   x 2048                46.4             74.2                83.3                89.8
    800   x 2048                63.9             85.9               122.9               132.9

Please note that dataset sizes refer to the shapes of the input arrays, which is kept the same for both of the inputs to ease the benchmarking process. Also, as the `non-MKL` setup, we had `OpenBLAS` for the matrix multiplications.

Quick conclusions could be drawn based upon the speedup figures -

- The performance boost with proposed methods scales proportionately to the dimensionality (number of columns)

- Leveraging `MKL` always proved to be better than `OpenBLAS`.
- For the GPU version, after computing the distances on GPU, it makes sense to  keep it on the GPU (if possible of course). This makes more sense with lesser dimensions cases.

For reference the system configuration used for benchmarking results had the specifications as listed below :

	Operating System: Ubuntu 16.04
	RAM: 16GB
	CPU Model: Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz (# Cores=4, # Threads=8)
	GPU Model : NVIDIA GeForce GTX 960M  (# CUDA Cores=640)

## Application examples

Few examples are run on IPython console.

### Example #1

```python
# Setup inputs
In [2]: import numpy as np
   ...: a = np.random.rand(10000,3)
   ...: b = np.random.rand(10000,3)

# Employ pairwise_distances
In [105]: from sklearn.metrics.pairwise import pairwise_distances

In [106]: %timeit pairwise_distances(a,b, 'sqeuclidean')
1 loop, best of 3: 282 ms per loop
```

Using proposed methods -

```python
# Import proposed CPU implementation and use it with best configuarion for such a dataset
In [111]: from eucl_dist.cpu_dist import dist

In [114]: %timeit dist(a, b, matmul='gemm', method='ext', precision='float32')
10 loops, best of 3: 131 ms per loop

# Import proposed GPU implementation
In [115]: from eucl_dist.gpu_dist import dist as gdist

# Use best configured GPU implementation with final output on CPU
In [117]: %timeit gdist(a, b, optimize_level=3)
10 loops, best of 3: 111 ms per loop

# Use best configured GPU implementation with final output kept on GPU
In [121]: %timeit gdist(a, b, optimize_level=3, output='gpu')
10 loops, best of 3: 21.5 ms per loop
```

### Example #2

Higher dimensionality case -

```python
In [238]: a = np.random.rand(800,2048).astype(np.float32)
     ...: b = np.random.rand(800,2048).astype(np.float32)

In [242]: %timeit pairwise_distances(a,b, 'sqeuclidean')
1 loop, best of 3: 922 ms per loop
```

With the best configurations from the proposed methods -

```python
In [253]: %timeit dist(a, b, matmul='dot', method='accum', precision='auto')
100 loops, best of 3: 14.3 ms per loop

In [255]: %timeit gdist(a, b, optimize_level=4) # Final output is back on CPU
100 loops, best of 3: 7.2 ms per loop
```

Speedup of **`64x+`** and **`128x+`** with the CPU and GPU based implementations respectively!

## Requirements

- Python 2.x or 3.x
- Python modules : NumPy, SciPy, Pandas, PyCUDA, scikit-cuda, scikit-learn
