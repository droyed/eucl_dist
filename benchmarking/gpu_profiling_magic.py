## This script profiles the GPU implementations with "%timeit" and saves
## the results as npy files for later review.

# Import modules and builtins
import pandas as pd
import itertools
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from eucl_dist.gpu_dist import dist as gdist

# Create function for sklearn built-in to compute squared euclidean distances
def sq_cdist(A,B): return pairwise_distances(A,B, 'sqeuclidean')

# Sets of inputs
M = np.array([[5000,3],[10000,3],[5000,20],[500,300],\
                          [1000,300],[500,2048],[800,2048]])

# Arguments
ops = [('cpu',0)] + list(itertools.product(["cpu","gpu"],range(1,5)))

l1, l2 = len(M), len(ops)
timings0 = np.zeros((l1))
timings1 = np.zeros((l1,l2))
for i,(m,n) in enumerate(M):
    A = np.random.randint(0,1000,(m,n)).astype(np.float32)
    B = np.random.randint(0,1000,(m,n)).astype(np.float32)

    # Profile built-in
    t_str = %timeit -oq sq_cdist(A,B)
    timings0[i] = t_str.best

    # Profile proposed method
    for j,(out_type, ol) in enumerate(ops):
        print(i,j)
        stmt1 = '{}(A, B,  ol, out_type)'.format('gdist')
        t_str = %timeit -oq eval(stmt1)
        timings1[i,j] = t_str.best

# Get speedups
speedups = timings0[:,None]/timings1

datasizes_str =[str(i).ljust(5)+' x '+str(j).rjust(4) for i,j in M]
input_args = [i+":"+str(j) for i,j in ops]
col_index = input_args
row_index = datasizes_str
speedup_df = pd.DataFrame(index=row_index, columns=col_index)
speedup_df[:] = speedups.reshape(speedup_df.shape)

# Get datraframe with best configurations for each dataset size and for
# CPU, GPU separately
from eucl_dist.util import save_df_as_npy, max_columns
cpu_speedup_df = speedup_df.loc[:,['cpu' in i for i in speedup_df.columns]]
gpu_speedup_df = speedup_df.loc[:,['gpu' in i for i in speedup_df.columns]]

best_cpu_df = max_columns(cpu_speedup_df, ['Config','Speedup'])
best_gpu_df = max_columns(gpu_speedup_df, ['Config','Speedup'])
best_config_df = pd.concat([best_cpu_df, best_gpu_df],axis=1, keys=['CPU','GPU'])

pd.set_option('precision', 1)
print("Speedups : \n")
print(speedup_df)
print("\nBest speedups :\n")
print(best_config_df)

# Save results for later study
save_df_as_npy("gpu_speedup_df.npy",speedup_df)
save_df_as_npy("gpu_best_speedup_df.npy",best_config_df)
