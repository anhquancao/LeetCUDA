from torch.utils.cpp_extension import load

lib = load(
    name="sigmoid_lib",
    sources=["sigmoid_me.cu"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    size = (13, 7, 512)
    x = torch.randn(size, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    print(x.shape)
    lib.sigmoid_f32(x, y)
    y_ref = torch.sigmoid(x)
    print("Max abs diff:", (y - y_ref).abs().max().item())
    print(y.shape)
    assert torch.allclose(y, y_ref, atol=1e-6), "Sigmoid kernel output does not match torch.sigmoid"
    print("Test passed.")
