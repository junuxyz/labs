import torch
import triton
import triton.language as tl

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # loads x from GPU RAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    output = x + y

    # after calculation, put output back to main memory
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(a: torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(a)
    n_elements = output.numel()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    add_kernel[grid](a,b,output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return output

size = 1024

a = torch.randn(size, device='cuda')
b = torch.randn(size, device='cuda')

output_triton = triton_add(a, b)

# calculate and compare with output_triton
output_pytorch = a + b

print("Triton Output:")
print(output_triton)
print("PyTorch Output:")
print(output_pytorch)

print(f"\nOutputs are close: {torch.allclose(output_triton, output_pytorch)}")