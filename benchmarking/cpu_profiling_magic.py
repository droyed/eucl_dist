## This script profiles the GPU implementations with "%timeit" and saves
## the results as npy files for later review.

# Import modules and builtins
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

# Import custom functions
from eucl_dist.cpu_dist import dist

# Create a function for the built-in to compute squared euclidean distances
def sq_cdist(A,B): return pairwise_distances(A,B, 'sqeuclidean')

# Sets of input defining sizes
M = np.array([[5000,3],[10000,3],[5000,20],[500,300],[1000,300],[500,2048],[800,2048]])

# Arguments to function : dist
matmuls = ["dot","gemm"]
methods = ["ext","accum"]
precisions = ["auto","float32"]

# Setup funcs for creating inputs of different dtypes
def rand_int (m): return np.random.randint(0,9,(m[0],m[1]))
def rand_f32 (m): return np.random.rand(*m).astype(np.float32)
def rand_f64 (m): return np.random.rand(*m).astype(np.float64)
funcs = [rand_int, rand_f32, rand_f64]
funcs_name = ['INT','F32','F64']

# Setup profiling :
# Arrays to store timings for builtin and proposed function.
l0,l1,l2,l3,l4 = len(funcs), len(M), len(matmuls), len(methods), len(precisions)
timings0 = np.zeros((l0,l1))
timings1 = np.zeros((l0,l1,l2,l3,l4))
# Start profiling
for h,f in enumerate(funcs):
    for i,m in enumerate(M):

        # Input arrays
        A = f(m)
        B = f(m)

        # Profile Scipy's built-in
        t_str = %timeit -oq sq_cdist(A,B)
        timings0[h,i] = t_str.best

        # Profile proposed method with three arguments with three loops
        for j,mm in enumerate(matmuls):
            for k,mt in enumerate(methods):
                for l,p in enumerate(precisions):
                    print(h,i,j,k,l)
                    stmt1 = '{}(A,B,mm,mt,p)'.format('dist')
                    t_str = %timeit -oq eval(stmt1)
                    timings1[h,i,j,k,l] = t_str.best

# Get speedups
speedups = timings0[:,:,None,None,None]/timings1

# Print all speedups as a multi-index dataframe for easy evaluation
np.set_printoptions(precision=2)
pd.set_option('precision', 2)
datasizes_str =[str(i).ljust(5)+' x '+str(j).rjust(4) for i,j in M]
input_args = [matmuls, methods, precisions]
col_index = pd.MultiIndex.from_product(input_args)
row_index = pd.MultiIndex.from_product([funcs_name,datasizes_str])

speedup_df = pd.DataFrame(index=row_index, columns=col_index)
speedup_df[:] = speedups.reshape(speedup_df.shape)

# Print best speedups as a multi-index dataframe for easy evaluation
arg_names = ['matmul','method','precision']
idx = speedups.reshape(speedups.shape[:2] + (-1,)).argmax(-1)
input_arg_lens = [len(i) for i in input_args]
argmax_idx = np.dstack((np.unravel_index(idx, input_arg_lens)))
best_args = np.array(input_args)[np.arange(argmax_idx.shape[-1]), argmax_idx]

best_speedup_df = pd.DataFrame(index=row_index, columns=arg_names)
best_speedup_df[:] = best_args.reshape(best_speedup_df.shape)
best_speedup_df['Speedup'] = speedups.max(axis=(-3,-2,-1)).ravel()

print("Speedups : ")
print(speedup_df)
print("\nBest speedups : ")
print(best_speedup_df)

# Save results for later study as npy files
import sys
if sys.version_info.major==3:
    choice = input("MKL yes or no?(y/n): ")
else:
    choice = raw_input("MKL yes or no?(y/n): ")

map1 = {'y':"" ,'n':"no"}
file_ext = map1[choice] + 'mkl_'

from eucl_dist.util import save_df_as_npy
np.save(file_ext+"timings0.npy", timings0)
save_df_as_npy(file_ext+"speedup_df.npy", speedup_df)
save_df_as_npy(file_ext+"best_speedup_df.npy", best_speedup_df)
