#include <math.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define MAX_EXP_F32 88.3762626647949f
#define MIN_EXP_F32 -88.3762626647949f

__global__ void sigmoid_f32_kernel(float* x, float* y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = x[idx];
        v = fminf(v, MAX_EXP_F32);
        v = fmaxf(v, MIN_EXP_F32);
        y[idx] = 1.0f/(1.0f+expf(-v));
    }
}


void sigmoid_f32(torch::Tensor x, torch::Tensor y) {
    const int ndim = x.dim();
    int N = 1;
    for (int i=0; i < ndim; i++) {
        N *= x.size(i);
    }
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((N+block_size-1)/block_size);
    sigmoid_f32_kernel<<<grid, block>>>(
        reinterpret_cast<float*>(x.data_ptr()),
        reinterpret_cast<float*>(y.data_ptr()),
        N
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_f32", &sigmoid_f32, "Sigmoid for float32");
}