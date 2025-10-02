import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


# triton internally finds the best combination for performance
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit  # tells us this code will be compiled
def add_kernel(
    x_ptr,  # Pointer to the first input vector.
    y_ptr,  # Pointer to the second input vector.
    output_ptr,  # Pointer to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Elements handled per program.
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # loads x and y from GPU RAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # after calculation, put output back to main memory
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(a)
    n_elements = output.numel()

    MAX_BLOCK_SIZE = 1024
    # calculates the ceiling division of n_elements to MAX_BLOCK_SIZE
    # to get the size of grid
    # the reason of cdiv is to keep every single data
    grid = (triton.cdiv(n_elements, MAX_BLOCK_SIZE),)

    add_kernel[grid](a, b, output, n_elements)

    return output


size = 1024

a = torch.randn(size, device="cuda")
b = torch.randn(size, device="cuda")

output_triton = triton_add(a, b)

# calculate and compare with output_triton
output_pytorch = a + b

print("Triton Output:")
print(output_triton)
print("PyTorch Output:")
print(output_pytorch)

print(f"\nOutputs are close: {torch.allclose(output_triton, output_pytorch)}")
