#include <math.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


__global__ void relu_f32_kernel(float *x, float *y, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(x[idx], 0.0f);
    }
}


void relu_f32(torch::Tensor x, torch::Tensor y) {
    const int ndim = x.dim();
    int N = 1;
    for (int i = 0; i < ndim; ++i) {
        N *= x.size(i);
    }
    int block_size = 256;
    dim3 block(block_size);
    dim3 grid((N + block_size - 1) / block_size);
    relu_f32_kernel<<<grid, block>>>(
        reinterpret_cast<float *>(x.data_ptr()),
        reinterpret_cast<float *>(y.data_ptr()), N);
}
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("relu_f32", &relu_f32, "Relu for float32");
}