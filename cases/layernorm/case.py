import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.case_util import get_torch_dtype, get_torch_device


class Case:
    def __init__(self):
        self.t_input: torch.Tensor = torch.tensor([])
        self.t_weight: torch.Tensor = torch.tensor([])
        self.t_bias: torch.Tensor = torch.tensor([])

    def save(self, path: str):
        tensors = {
            "input": self.t_input,
            "weight": self.t_weight,
            "bias": self.t_bias,
        }
        torch.save(tensors, path)

    def load(self, path: str):
        tensors = torch.load(path)
        self.t_input = tensors["input"]
        self.t_weight = tensors["weight"]
        self.t_bias = tensors["bias"]


class Result:
    def __init__(self):
        self.t_output: torch.Tensor = torch.tensor([])

    def save(self, path: str):
        tensors = {"output": self.t_output}
        torch.save(tensors, path)

    def load(self, path: str):
        tensors = torch.load(path)
        self.t_output = tensors["output"]

    def get_tensor_list(self) -> list[torch.Tensor]:
        return [self.t_output]


def caller(case: Case, dtype, device) -> Result:
    torch_device = get_torch_device(device)
    torch_dtype = get_torch_dtype(dtype)
    if device == "baseline":
        torch_dtype = torch.float64

    result = Result()
    input_tensor = case.t_input.to(device=torch_device, dtype=torch_dtype)
    result.t_output = torch.nn.functional.layer_norm(input_tensor, input_tensor.shape[-1:], case.t_weight.to(device=torch_device, dtype=torch_dtype), case.t_bias.to(device=torch_device, dtype=torch_dtype))
    return result