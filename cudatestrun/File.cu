#include "cuda_runtime.h"
#include "./header.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>
using namespace std::chrono;

#define N 100000000
#define N 100000000
#define NUMTHREADS 512

//#define N 10000
//#define NUMTHREADS 100


__device__ bool isprime(int n)
{
    if (n <= 1)
        return false;
    int s = sqrt((float)n);

    for (int i = 2; i <= s; ++i)
    {
        if (n % i == 0)
            return false;
    }
    return true;
}

__global__ void kernel_calc_primes_dev(int* a,int* primecount)
{
    __shared__ float cache[NUMTHREADS];
    int tid = threadIdx.x;
    int count = 0;
    while (tid < N)
    {
        if (isprime(a[tid]))
            ++count;
        tid += NUMTHREADS;
    }

    cache[threadIdx.x] = count;
    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < NUMTHREADS;++i)
        {
            *primecount += cache[i];
        }
    }
}

void kernel_calc_primes_host()
{
    int c;
    int *dev_c;
    int* a = new int[N];
    for (int i = 0; i < N; ++i)
    {
        a[i] = i + 1;
    }
    int* dev_a;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    cudaMalloc((void**)&dev_c, sizeof(int));
    cudaMalloc((void**)&dev_a, sizeof(int) * N);

    cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyKind::cudaMemcpyHostToDevice);

    kernel_calc_primes_dev <<< 1, NUMTHREADS >>> (dev_a,dev_c);

    cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cudaFree(dev_a);
    cudaFree(dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("%d primes found between [%d] and [%d] in %d seconds.", c,1,N, (int)elapsedTime/1000);

    delete[] a;

}

