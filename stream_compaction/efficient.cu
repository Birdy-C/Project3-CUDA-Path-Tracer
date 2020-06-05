#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void dev_scan(int n, int *g_odata, const int *g_idata) {
            extern __shared__ int temp[]; // allocated on invocation   
            int thid = (blockIdx.x * blockDim.x) + threadIdx.x;
            int offset = 1; 
            if (thid * 2 >= n) {
                return;
            }

            // load input into shared memory
            temp[2 * thid] = g_idata[2 * thid];
            temp[2 * thid + 1] = g_idata[2 * thid + 1];

            // build sum in place up the tree
            for (int d = n >> 1; d > 0; d >>= 1)
            {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            // clear the last element  
            if (thid == 0) 
            {
                temp[n - 1] = 0;
            } 
            
            // traverse down tree & build scan 
            for (int d = 1; d < n; d *= 2) 
            {
                offset >>= 1;
                __syncthreads();
                if (thid < d)
                {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;
                    float t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }
            __syncthreads();
            g_odata[2 * thid] = temp[2 * thid]; // write results to device memory   
            g_odata[2 * thid + 1] = temp[2 * thid + 1]; 
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_odata;
            int ceiln = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_idata, ceiln * sizeof(int));
            cudaMalloc((void**)&dev_odata, ceiln * sizeof(int));
            cudaMemset(dev_idata, 0, ceiln * sizeof(int));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            dev_scan << <1, ceiln / 2, ceiln * sizeof(int) >> > (ceiln, dev_odata, dev_idata);
            timer().endGpuTimer();
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int *dev_idata;
            int *dev_bools;
            int *dev_indices;
            int *dev_odata;
            int ceiln = 1 << ilog2ceil(n);
            cudaMalloc((void**)&dev_idata, ceiln * sizeof(int));
            cudaMalloc((void**)&dev_bools, ceiln * sizeof(int));
            cudaMalloc((void**)&dev_indices, ceiln * sizeof(int));
            cudaMalloc((void**)&dev_odata, ceiln * sizeof(int));
            cudaMemset(dev_idata, 0, ceiln * sizeof(int));
            cudaMemcpy(dev_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            // TODO
            StreamCompaction::Common::kernMapToBoolean<< <1, ceiln >> > (ceiln, dev_bools, dev_idata);
            dev_scan << <1, ceiln / 2, ceiln * sizeof(int) >> > (ceiln, dev_indices, dev_bools);
            StreamCompaction::Common::kernScatter << <1, n >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();
            int resulsize;
            cudaMemcpy(odata, dev_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            cudaMemcpy(&resulsize, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            return resulsize + (idata[n - 1] != 0);
        }
    }
}
