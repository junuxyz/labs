import torch

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')

size = 1024

a = torch.randn(size, device = 'cuda')
b = torch.randn(size, device = 'cuda')

# torch.empty() is a bit faster than torch.zeros() or torch.randn() 
# because it just uses garbage value from its memory space
output = torch.empty(size, device = 'cuda')

output = a + b

print("PyTorch Output:")
print(output)