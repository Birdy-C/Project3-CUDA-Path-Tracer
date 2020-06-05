#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::host_vector<int> H_in(idata, idata + n);
            thrust::device_vector<int> D_in = H_in;
            thrust::host_vector<int> H_out(idata, idata + n);
            thrust::device_vector<int> D_out = H_out;
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
            thrust::exclusive_scan(D_in.begin(), D_in.end(), D_out.begin());
            timer().endGpuTimer();
            thrust::copy(D_out.begin(), D_out.end(), odata);
        }
    }
}
