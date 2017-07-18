## This script loads the dataframes from NPY files saved earlier with "%timeit"
## command for profiling and prints those in pretty dataframes for easy viewing

# Import modules and builtins
import numpy as np
import pandas as pd
from eucl_dist.util import load_df_from_npy, max_columns as maxc

# Print options for readable figures
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:3,.2f}'.format

## PART-1 : No MKL speedups
print("----------------------------- No MKL ----------------------------- \n")
# Read in figures from saved files
nomkl_timings0 = np.load("nomkl_timings0.npy")
nomkl_speedup_df = load_df_from_npy("nomkl_speedup_df.npy")
nomkl_best_speedup_df = load_df_from_npy("nomkl_best_speedup_df.npy")

print("Speedups : \n")
print(nomkl_speedup_df)
print("\nBest speedups :\n")
print(nomkl_best_speedup_df)

## PART-2 : With MKL speedups
print("\n\n----------------------------- MKL ----------------------------- \n")
mkl_timings0 = np.load("mkl_timings0.npy")
mkl_speedup_df = load_df_from_npy("mkl_speedup_df.npy")
mkl_best_speedup_df = load_df_from_npy("mkl_best_speedup_df.npy")

shp = [len(list(i)) for i in mkl_speedup_df.index.levels] + [-1,]
nomkl_timings1 = nomkl_timings0[...,None]/nomkl_speedup_df.values.reshape(shp)
mkl_timings1 = mkl_timings0[...,None]/mkl_speedup_df.values.reshape(shp)

mkl_speedup_over_nomkl = nomkl_timings0[...,None]/mkl_timings1
mkl_speedup_over_nomkl_df = mkl_speedup_df.copy()
mkl_speedup_over_nomkl_df[:] = mkl_speedup_over_nomkl.reshape(mkl_speedup_over_nomkl_df.shape)

cols = ['matmul','method','precision', 'Speedup']
best_mkl_speedup_over_nomkl_df = maxc(mkl_speedup_over_nomkl_df, cols)

print("Speedups : \n")
print(mkl_speedup_over_nomkl_df)
print("\nBest speedups :\n")
print(best_mkl_speedup_over_nomkl_df)

## PART-3 : GPU speedups
print("\n\n----------------------------- GPU ----------------------------- \n")

gpu_df = load_df_from_npy("gpu_speedup_df.npy")
gpu_best_config_df = load_df_from_npy("gpu_best_speedup_df.npy")

pd.options.display.float_format = '{:3,.1f}'.format
print("Speedups : \n")
print(gpu_df)
print("\nBest speedups :\n")
print(gpu_best_config_df)

## PART-4 : Combined best speedups
print("\n\n---------------------- Combined best -------------------------- \n")

maxc_comb = [maxc(nomkl_speedup_df,cols), maxc(mkl_speedup_over_nomkl_df, cols)]
cpu_best_config_df = pd.concat(maxc_comb, axis=1, keys=['noMKL','MKL'])
all_best_config_df = pd.concat([cpu_best_config_df.loc['F32'], gpu_best_config_df],axis=1)
print("\nBest configurations :\n")
print all_best_config_df
