import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.case_util import save_all_cases_by_name
from cases.softmax.case import Case


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    case_list = []

    # Case 1: Normal range values (typical attention logits)
    case1 = Case()
    case1.t_input = torch.randn(4, 128) * 5
    case_list.append(case1)

    # Case 2: Large positive values (overflow risk)
    case2 = Case()
    case2.t_input = torch.rand(4, 128) * 50 + 50
    case_list.append(case2)

    # Case 3: Large negative values (underflow risk)
    case3 = Case()
    case3.t_input = torch.rand(4, 128) * (-50) - 50
    case_list.append(case3)

    # Case 4: Mixed extreme values
    case4 = Case()
    case4.t_input = torch.randn(4, 128) * 50
    case_list.append(case4)

    # Case 5: Very large positive values (fp32 overflow)
    case5 = Case()
    case5.t_input = torch.rand(4, 128) * 40 + 80
    case_list.append(case5)

    # Case 6: Near fp32 max (3.4e38)
    case6 = Case()
    case6.t_input = torch.tensor([[1e30, 1e31, 1e32, 1e33]] * 32)
    case_list.append(case6)

    # Case 7: One very large value with small values
    case7 = Case()
    case7.t_input = torch.ones(4, 128) * (-10)
    case7.t_input[:, 0] = 50
    case_list.append(case7)

    # Case 8: All equal values (numerical stability)
    case8 = Case()
    case8.t_input = torch.ones(4, 128) * 5.0
    case_list.append(case8)

    # Case 9: Very small differences (precision test)
    case9 = Case()
    base = torch.ones(4, 128) * 10.0
    noise = torch.randn(4, 128) * 1e-6
    case9.t_input = base + noise
    case_list.append(case9)

    # Case 10: Large sequence length (typical for LLM)
    case10 = Case()
    case10.t_input = torch.randn(2, 2048) * 10
    case_list.append(case10)

    # Case 11: Alternating large/small values
    case11 = Case()
    case11.t_input = torch.zeros(4, 128)
    case11.t_input[:, ::2] = 50  # Even positions
    case11.t_input[:, 1::2] = -50  # Odd positions
    case_list.append(case11)

    # Case 12: Exponential distribution (common in attention)
    case12 = Case()
    case12.t_input = torch.exp(torch.linspace(-10, 0, 128)).repeat(4, 1)
    case_list.append(case12)

    # Case 13: Near-zero values
    case13 = Case()
    case13.t_input = torch.randn(4, 128) * 1e-3
    case_list.append(case13)

    # Case 14: Large batch with typical attention pattern
    case14 = Case()
    batch_size, seq_len = 16, 512
    case14.t_input = torch.randn(batch_size, seq_len) * 5
    # Add causal mask effect (large negative values)
    for i in range(batch_size):
        mask_end = torch.randint(seq_len//2, seq_len, (1,)).item()
        case14.t_input[i, mask_end:] = -1e4
    case_list.append(case14)

    # Case 15: Values near fp32 precision limit
    case15 = Case()
    case15.t_input = torch.tensor([[16777216.0, 16777217.0, 16777218.0, 16777220.0]] * 32)
    case_list.append(case15)

    # Case 16: Gradient-like values (small)
    case16 = Case()
    case16.t_input = torch.rand(4, 128) * 0.01 + 1e-4
    case_list.append(case16)

    # Save the cases to the specified path
    save_all_cases_by_name(case_list, "softmax", "fp32")
    print(f"Successfully generated {len(case_list)} test cases")
