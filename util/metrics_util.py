import torch


def count_inf(tensor: torch.Tensor) -> int:
    return torch.isinf(tensor).sum().item()

def count_nan(tensor: torch.Tensor) -> int:
    return torch.isnan(tensor).sum().item()

def count_zero(tensor: torch.Tensor) -> int:
    return (tensor == 0).sum().item()

def get_abs_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    t1 = tensor1.to(dtype=torch.float64, device='cpu')
    t2 = tensor2.to(dtype=torch.float64, device='cpu')
    return torch.abs(t1 - t2)

def get_max_abs_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    abs_error = get_abs_error(tensor1, tensor2)
    return torch.max(abs_error).item()

def get_avg_abs_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    abs_error = get_abs_error(tensor1, tensor2)
    return torch.mean(abs_error).item()

def get_rel_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    t1 = tensor1.to(dtype=torch.float64, device='cpu')
    t2 = tensor2.to(dtype=torch.float64, device='cpu')
    return torch.abs(t1 - t2) / (torch.abs(t2) + 1e-7)

def get_max_rel_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    rel_error = get_rel_error(tensor1, tensor2)
    return torch.max(rel_error).item()

def get_avg_rel_error(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    rel_error = get_rel_error(tensor1, tensor2)
    return torch.mean(rel_error).item()