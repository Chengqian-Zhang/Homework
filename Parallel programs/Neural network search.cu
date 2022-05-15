#include <stdio.h>
#include <cuda.h>
    __constant__ float c[19] = { 0.0,0.0,0.0,0.0,2.0, 0.0,0.0,0.0,0.0,2.0, 0.0,0.0,0.0,0.0,2.0, -2.0,2.0,-2.0,4.0 }，add[4] = { -1.5f,-0.5f,0.5f,1.5f }, last[4] = { 2.5f,3.5f,4.5f,5.5f };
    __constant__ int ans[16] = { 0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0 };
    __device__ float relu(float x) { return x * (x > 0.0f); }
    __global__ void xor_kernel() {
        unsigned long long id = (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
        float w[19]; for (int i = 0; i < 18; i++) w[i] = c[i] + add[(id >> (2 * i)) % 4];
        int err[4] = { 0,0,0,0 };
        for (int i = 0; i < 4; i++) {
            int j = 0;
            for (int x = 0; x < 16; x++) {
                bool x3 = x & 1, x2 = (x >> 1) & 1, x1 = (x >> 2) & 1, x0 = (x >> 3) & 1; bool y = ans[j]; j++;
                float a0 = relu(x0 * w[0] + x1 * w[1] + x2 * w[2] + x3 * w[3] + w[4]),
                    a1 = relu(x0 * w[5] + x1 * w[6] + x2 * w[7] + x3 * w[8] + w[9]),
                    a2 = relu(x0 * w[10] + x1 * w[11] + x2 * w[12] + x3 * w[13] + w[14]);
                float z = a0 * w[15] + a1 * w[16] + a2 * w[17] + last[i];
                err[i] += ((z >= 0) != y);
            }
        }
        if (err[0] == 0) printf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
            w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16], w[17], last[0]);
        if (err[1] == 0) printf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
            w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16], w[17], last[1]);
        if (err[2] == 0) printf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
            w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16], w[17], last[2]);
        if (err[3] == 0) printf("%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f\n",
            w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15], w[16], w[17], last[3]);
    }
    int main(void) {
        xor_kernel << <512, 1024 >> > ();
        cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
        xor_kernel << < 1 << 29, 128 >> > ();
        cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
        float elapsed_time = 0.0f; cudaEventElapsedTime(&elapsed_time, start, stop);
        printf("Kernel Elapsed Time: %.3f s\n", elapsed_time / 1000.0);
        return 0;
    }
