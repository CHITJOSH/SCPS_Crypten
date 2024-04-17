import crypten
import torch

crypten.init()

a = torch.randn(3, 3) * 5
b = torch.randn(3, 3) * 5

result = a.add(crypten.cryptensor(b))
print(result)
reference = a.add(b)
print(reference)
assert torch.allclose(reference, result.get_plain_text(), atol=2e-4)

result = a.sub(crypten.cryptensor(b))
print(result)
reference = a.sub(b)
print(reference)
assert torch.allclose(reference, result.get_plain_text(), atol=2e-4)

result = a.mul(crypten.cryptensor(b))
reference = a.mul(b)
print(reference)
assert torch.allclose(reference, result.get_plain_text(), atol=2e-4)