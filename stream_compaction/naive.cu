#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void dev_scan(int n, int *g_odata, const int *g_idata) {
            extern __shared__ int temp[]; // allocated on invocation   
            int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
            int pout = 0, pin = 1;
            if (thid >= n) {
                return;
            }
            // Load input into shared memory.
            // This is exclusive scan, so shift right by one   
            // and set first element to 0   
            temp[pout*n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;
            __syncthreads();
            for (int offset = 1; offset < n; offset *= 2)
            {
                pout = 1 - pout; // swap double buffer indices  
                pin = 1 - pout;
                if (thid >= offset)
                    temp[pout*n + thid] = temp[pin*n + thid] + temp[pin*n + thid - offset];
                else  
                    temp[pout*n + thid] = temp[pin*n + thid];
                __syncthreads();
            }

            g_odata[thid] = temp[pout*n + thid]; // write output 
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_odata;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            dev_scan <<<1, n, 2 * n * sizeof(int)>>> (n, dev_odata, dev_idata);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }
    }
}
