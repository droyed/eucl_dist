## This script basically encapsulates relevant CUDA kernel codes as
## blocks of strings for use with GPU implementations to get euclidean distances.

sq_sum_codetext = """
__global__ void sq_sum(float *dest, float *dest2, float *a, float *b, int* SZ)
{
    // Store parameters
    const int M = SZ[0];
    const int N = SZ[1];
    const int M2 = SZ[2];
    const int t = threadIdx.x;
    const int TBP = blockDim.x;
    const int bx = blockIdx.x;
    const int GSZ = gridDim.x;

    // Initialize iterators
    int i,I,j,inv;

    // Initialize 2D shaped memory. The first axis stores the summations for
    // each row. The second axis is the column and data gets reduced along it.
    __shared__ float aS[BLOCK_SIZE][GRID_SIZE];

    // Loop to square sum-reduce each row of first array
    for(i=0;i<M/GSZ;i++)
    {
        j = bx;
        I = i*GSZ+bx;

        inv = TBP;
        aS[t][j] = a[I*N+t]*a[I*N+t];
        __syncthreads();

        if(t+inv<N)
            aS[t][j] += a[I*N+t+inv]*a[I*N+t+inv];
        __syncthreads();

        inv = inv/2;
        while(inv!=0)
        {
            if(t<inv)
                aS[t][j] += aS[t+inv][j];
            __syncthreads();
            inv = inv/2;
        }
        __syncthreads();
        if(t==0)
            dest[I] = aS[0][j];
        __syncthreads();
    }


    // Loop to square sum-reduce each row of second array
    for(i=0;i<M2/GSZ;i++)
    {
        j = bx;
        I = i*GSZ+bx;

        inv = TBP;
        aS[t][j] = b[I*N+t]*b[I*N+t];
        __syncthreads();

        if(t+inv<N)
            aS[t][j] += b[I*N+t+inv]*b[I*N+t+inv];
        __syncthreads();

        inv = inv/2;
        while(inv!=0)
        {
            if(t<inv)
                aS[t][j] += aS[t+inv][j];
            __syncthreads();
            inv = inv/2;
        }
        __syncthreads();
        if(t==0)
            dest2[I] = aS[0][j];
        __syncthreads();
    }
}
"""

addvecs_codetext = """
__global__ void add_vectors_broadcast(float *dest, float *a, float *b, int* SZ)
{
    const int M = SZ[0];
    const int N = SZ[1];
    const int S = SZ[2];

    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int BSZ = blockDim.x;
    int t;

    for (int s=0;s<S;s++)
    {
        t = s*BSZ+tx;
        if(t<N)
            dest[bx*N+t] = b[t] + a[bx];
        __syncthreads();
    }

}
"""
