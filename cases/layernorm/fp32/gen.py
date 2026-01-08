import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from util.case_util import save_all_cases_by_name
from cases.layernorm.case import Case


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    case_list = []
    hidden_size = 768  # Typical LLM hidden size

    # Helper function to create default weight and bias
    def create_params(size):
        weight = torch.ones(size)
        bias = torch.zeros(size)
        return weight, bias

    # Case 1: Normal distribution (typical activations)
    case1 = Case()
    case1.t_input = torch.randn(4, 128, hidden_size)
    case1.t_weight, case1.t_bias = create_params(hidden_size)
    case_list.append(case1)

    # Case 2: Large positive values
    case2 = Case()
    case2.t_input = torch.rand(4, 128, hidden_size) * 50 + 50
    case2.t_weight, case2.t_bias = create_params(hidden_size)
    case_list.append(case2)

    # Case 3: Large negative values
    case3 = Case()
    case3.t_input = torch.rand(4, 128, hidden_size) * (-50) - 50
    case3.t_weight, case3.t_bias = create_params(hidden_size)
    case_list.append(case3)

    # Case 4: Mixed extreme values
    case4 = Case()
    case4.t_input = torch.randn(4, 128, hidden_size) * 50
    case4.t_weight, case4.t_bias = create_params(hidden_size)
    case_list.append(case4)

    # Case 5: Very small variance (numerical stability challenge)
    case5 = Case()
    base = torch.ones(4, 128, hidden_size) * 10.0
    noise = torch.randn(4, 128, hidden_size) * 1e-7
    case5.t_input = base + noise
    case5.t_weight, case5.t_bias = create_params(hidden_size)
    case_list.append(case5)

    # Case 6: Near-zero values
    case6 = Case()
    case6.t_input = torch.randn(4, 128, hidden_size) * 1e-3
    case6.t_weight, case6.t_bias = create_params(hidden_size)
    case_list.append(case6)

    # Case 7: All equal values (zero variance)
    case7 = Case()
    case7.t_input = torch.ones(4, 128, hidden_size) * 5.0
    case7.t_weight, case7.t_bias = create_params(hidden_size)
    case_list.append(case7)

    # Case 8: Large hidden dimension (4096 for large LLMs)
    case8 = Case()
    large_hidden = 4096
    case8.t_input = torch.randn(2, 64, large_hidden) * 2
    case8.t_weight, case8.t_bias = create_params(large_hidden)
    case_list.append(case8)

    # Case 9: Values near fp32 precision limit
    case9 = Case()
    case9.t_input = torch.ones(4, 128, hidden_size) * 16777216.0 + torch.randn(4, 128, hidden_size)
    case9.t_weight, case9.t_bias = create_params(hidden_size)
    case_list.append(case9)

    # Case 10: Non-unit weight and bias
    case10 = Case()
    case10.t_input = torch.randn(4, 128, hidden_size) * 5
    case10.t_weight = torch.randn(hidden_size) * 2 + 1
    case10.t_bias = torch.randn(hidden_size) * 0.5
    case_list.append(case10)

    # Case 11: Extreme weight values
    case11 = Case()
    case11.t_input = torch.randn(4, 128, hidden_size)
    case11.t_weight = torch.ones(hidden_size) * 100
    case11.t_bias = torch.ones(hidden_size) * 50
    case_list.append(case11)

    # Case 12: Very small weight values
    case12 = Case()
    case12.t_input = torch.randn(4, 128, hidden_size) * 10
    case12.t_weight = torch.ones(hidden_size) * 1e-5
    case12.t_bias = torch.ones(hidden_size) * 1e-5
    case_list.append(case12)

    # Case 13: Outliers in input
    case13 = Case()
    case13.t_input = torch.randn(4, 128, hidden_size)
    case13.t_input[0, 0, 0] = 1000.0  # Outlier
    case13.t_input[1, 10, 10] = -1000.0  # Outlier
    case13.t_weight, case13.t_bias = create_params(hidden_size)
    case_list.append(case13)

    # Case 14: Gradient-like small values
    case14 = Case()
    case14.t_input = torch.rand(4, 128, hidden_size) * 0.01 + 1e-4
    case14.t_weight, case14.t_bias = create_params(hidden_size)
    case_list.append(case14)

    # Case 15: Long sequence (typical for LLM)
    case15 = Case()
    case15.t_input = torch.randn(2, 2048, hidden_size) * 3
    case15.t_weight, case15.t_bias = create_params(hidden_size)
    case_list.append(case15)

    # Case 16: Exponential distribution
    case16 = Case()
    case16.t_input = torch.randn(4, 128, hidden_size).abs() * 10
    case16.t_weight, case16.t_bias = create_params(hidden_size)
    case_list.append(case16)

    # Case 17: Alternating high/low variance across features
    case17 = Case()
    case17.t_input = torch.randn(4, 128, hidden_size)
    case17.t_input[:, :, :hidden_size//2] *= 0.01  # Low variance
    case17.t_input[:, :, hidden_size//2:] *= 10.0  # High variance
    case17.t_weight, case17.t_bias = create_params(hidden_size)
    case_list.append(case17)

    # Case 18: Very large values (overflow risk)
    case18 = Case()
    case18.t_input = torch.rand(4, 128, hidden_size) * 9e6 + 1e6
    case18.t_weight, case18.t_bias = create_params(hidden_size)
    case_list.append(case18)

    # Case 19: Negative weight (unusual but possible)
    case19 = Case()
    case19.t_input = torch.randn(4, 128, hidden_size) * 5
    case19.t_weight = -torch.ones(hidden_size) * 2
    case19.t_bias = torch.zeros(hidden_size)
    case_list.append(case19)

    # Case 20: Single batch dimension
    case20 = Case()
    case20.t_input = torch.randn(1, 512, hidden_size) * 2
    case20.t_weight, case20.t_bias = create_params(hidden_size)
    case_list.append(case20)

    # Save the cases to the specified path
    save_all_cases_by_name(case_list, "layernorm", "fp32")
    print(f"Successfully generated {len(case_list)} test cases")
