import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import sys
import random
from datetime import datetime

# Set seeds for deterministic behavior
def set_deterministic_mode(seed: int = 42):
    """Set all random seeds and configure for deterministic execution"""
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        
        # CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Additional deterministic settings
    torch.use_deterministic_algorithms(True, warn_only=True)

# Initialize deterministic mode at import time
set_deterministic_mode(42)

# Ensure CUDA is available for cuDNN tests
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. GPU tests will be skipped.")

class TeeLogger:
    """Redirect stdout to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', buffering=1)  # Line buffering
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

class LayerNormPrecisionTester:
    """Test LayerNorm precision with fp32 vs fp64 ground truth"""
    
    def __init__(self):
        self.results = []
        
    def compute_metrics(self, ground_truth: torch.Tensor, test_result: torch.Tensor, 
                       test_name: str, case_name: str) -> Dict:
        """Compute precision metrics between ground truth and test result"""
        
        # Convert to numpy for easier computation
        gt = ground_truth.cpu().numpy().astype(np.float64)
        test = test_result.cpu().numpy().astype(np.float64)
        
        # Absolute error
        abs_error = np.abs(gt - test)
        max_abs_error = np.max(abs_error)
        mean_abs_error = np.mean(abs_error)
        
        # Relative error (avoid division by zero)
        epsilon = 1e-20
        rel_error = np.abs((gt - test) / (np.abs(gt) + epsilon))
        max_rel_error = np.max(rel_error)
        mean_rel_error = np.mean(rel_error)
        
        # Count inf and NaN
        inf_count = np.isinf(test).sum()
        nan_count = np.isnan(test).sum()
        
        metrics = {
            'test_name': test_name,
            'case_name': case_name,
            'max_abs_error': float(max_abs_error),
            'mean_abs_error': float(mean_abs_error),
            'max_rel_error': float(max_rel_error),
            'mean_rel_error': float(mean_rel_error),
            'inf_count': int(inf_count),
            'nan_count': int(nan_count),
            'ground_truth_inf': int(np.isinf(gt).sum()),
            'ground_truth_nan': int(np.isnan(gt).sum()),
        }
        
        return metrics
    
    def layernorm_fp64_cpu(self, x: torch.Tensor, normalized_shape: Tuple, 
                           weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Ground truth: LayerNorm with fp64 on CPU"""
        x_fp64 = x.to(dtype=torch.float64, device='cpu')
        weight_fp64 = weight.to(dtype=torch.float64, device='cpu') if weight is not None else None
        bias_fp64 = bias.to(dtype=torch.float64, device='cpu') if bias is not None else None
        return torch.nn.functional.layer_norm(x_fp64, normalized_shape, weight_fp64, bias_fp64, eps=1e-5)
    
    def layernorm_fp32_cpu(self, x: torch.Tensor, normalized_shape: Tuple,
                           weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Test 1: LayerNorm with fp32 on CPU"""
        x_fp32 = x.to(dtype=torch.float32, device='cpu')
        weight_fp32 = weight.to(dtype=torch.float32, device='cpu') if weight is not None else None
        bias_fp32 = bias.to(dtype=torch.float32, device='cpu') if bias is not None else None
        return torch.nn.functional.layer_norm(x_fp32, normalized_shape, weight_fp32, bias_fp32, eps=1e-5)
    
    def layernorm_fp32_gpu(self, x: torch.Tensor, normalized_shape: Tuple,
                           weight: torch.Tensor = None, bias: torch.Tensor = None) -> torch.Tensor:
        """Test 2: LayerNorm with fp32 on GPU (cuDNN)"""
        if not torch.cuda.is_available():
            return None
        x_fp32 = x.to(dtype=torch.float32, device='cuda')
        weight_fp32 = weight.to(dtype=torch.float32, device='cuda') if weight is not None else None
        bias_fp32 = bias.to(dtype=torch.float32, device='cuda') if bias is not None else None
        result = torch.nn.functional.layer_norm(x_fp32, normalized_shape, weight_fp32, bias_fp32, eps=1e-5)
        return result
    
    def generate_test_cases(self) -> List[Tuple[str, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate comprehensive test cases for LLM training scenarios
        Returns: List of (case_name, input, weight, bias)
        """
        test_cases = []
        hidden_size = 768  # Typical LLM hidden size
        
        # Create default weight and bias
        def create_params(size):
            weight = torch.ones(size)
            bias = torch.zeros(size)
            return weight, bias
        
        # Case 1: Normal distribution (typical activations)
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Normal distribution N(0,1)",
            torch.randn(4, 128, hidden_size),
            weight, bias
        ))
        
        # Case 2: Large positive values
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Large positive [50, 100]",
            torch.rand(4, 128, hidden_size) * 50 + 50,
            weight, bias
        ))
        
        # Case 3: Large negative values
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Large negative [-100, -50]",
            torch.rand(4, 128, hidden_size) * (-50) - 50,
            weight, bias
        ))
        
        # Case 4: Mixed extreme values
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Mixed extremes [-100, 100]",
            torch.randn(4, 128, hidden_size) * 50,
            weight, bias
        ))
        
        # Case 5: Very small variance (numerical stability challenge)
        weight, bias = create_params(hidden_size)
        base = torch.ones(4, 128, hidden_size) * 10.0
        noise = torch.randn(4, 128, hidden_size) * 1e-7
        test_cases.append((
            "Small variance (1e-7)",
            base + noise,
            weight, bias
        ))
        
        # Case 6: Near-zero values
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Near-zero [-1e-3, 1e-3]",
            torch.randn(4, 128, hidden_size) * 1e-3,
            weight, bias
        ))
        
        # Case 7: All equal values (zero variance)
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "All equal (zero variance)",
            torch.ones(4, 128, hidden_size) * 5.0,
            weight, bias
        ))
        
        # Case 8: Large hidden dimension (4096 for large LLMs)
        large_hidden = 4096
        weight, bias = create_params(large_hidden)
        test_cases.append((
            "Large hidden size [4096]",
            torch.randn(2, 64, large_hidden) * 2,
            weight, bias
        ))
        
        # Case 9: Values near fp32 precision limit
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Near fp32 precision limit",
            torch.ones(4, 128, hidden_size) * 16777216.0 + torch.randn(4, 128, hidden_size),
            weight, bias
        ))
        
        # Case 10: Non-unit weight and bias
        weight = torch.randn(hidden_size) * 2 + 1
        bias = torch.randn(hidden_size) * 0.5
        test_cases.append((
            "Non-trivial weight/bias",
            torch.randn(4, 128, hidden_size) * 5,
            weight, bias
        ))
        
        # Case 11: Extreme weight values
        weight = torch.ones(hidden_size) * 100
        bias = torch.ones(hidden_size) * 50
        test_cases.append((
            "Extreme weight/bias",
            torch.randn(4, 128, hidden_size),
            weight, bias
        ))
        
        # Case 12: Very small weight values
        weight = torch.ones(hidden_size) * 1e-5
        bias = torch.ones(hidden_size) * 1e-5
        test_cases.append((
            "Small weight/bias (1e-5)",
            torch.randn(4, 128, hidden_size) * 10,
            weight, bias
        ))
        
        # Case 13: Outliers in input
        weight, bias = create_params(hidden_size)
        case13 = torch.randn(4, 128, hidden_size)
        case13[0, 0, 0] = 1000.0  # Outlier
        case13[1, 10, 10] = -1000.0  # Outlier
        test_cases.append((
            "Input with outliers",
            case13,
            weight, bias
        ))
        
        # Case 14: Gradient-like small values
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Gradient-like [1e-4, 1e-2]",
            torch.rand(4, 128, hidden_size) * 0.01 + 1e-4,
            weight, bias
        ))
        
        # Case 15: Long sequence (typical for LLM)
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Long sequence [2048]",
            torch.randn(2, 2048, hidden_size) * 3,
            weight, bias
        ))
        
        # Case 16: Exponential distribution
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Exponential distribution",
            torch.randn(4, 128, hidden_size).abs() * 10,
            weight, bias
        ))
        
        # Case 17: Alternating high/low variance across features
        weight, bias = create_params(hidden_size)
        case17 = torch.randn(4, 128, hidden_size)
        case17[:, :, :hidden_size//2] *= 0.01  # Low variance
        case17[:, :, hidden_size//2:] *= 10.0  # High variance
        test_cases.append((
            "Mixed feature variance",
            case17,
            weight, bias
        ))
        
        # Case 18: Very large values (overflow risk)
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Very large [1e6, 1e7]",
            torch.rand(4, 128, hidden_size) * 9e6 + 1e6,
            weight, bias
        ))
        
        # Case 19: Negative weight (unusual but possible)
        weight = -torch.ones(hidden_size) * 2
        bias = torch.zeros(hidden_size)
        test_cases.append((
            "Negative weight",
            torch.randn(4, 128, hidden_size) * 5,
            weight, bias
        ))
        
        # Case 20: Single batch dimension
        weight, bias = create_params(hidden_size)
        test_cases.append((
            "Single batch [1, 512, 768]",
            torch.randn(1, 512, hidden_size) * 2,
            weight, bias
        ))
        
        return test_cases
    
    def run_single_test(self, case_name: str, x: torch.Tensor, 
                       weight: torch.Tensor, bias: torch.Tensor):
        """Run a single test case"""
        normalized_shape = (x.shape[-1],)  # Normalize over last dimension
        
        print(f"\n{'='*80}")
        print(f"Test Case: {case_name}")
        print(f"Shape: {x.shape}, Range: [{x.min().item():.4e}, {x.max().item():.4e}]")
        print(f"Mean: {x.mean().item():.4e}, Std: {x.std().item():.4e}")
        print(f"Weight shape: {weight.shape}, Bias shape: {bias.shape}")
        print(f"{'='*80}")
        
        # Ground truth: fp64 on CPU
        ground_truth = self.layernorm_fp64_cpu(x, normalized_shape, weight, bias)
        
        # Test 1: fp32 on CPU
        test1_result = self.layernorm_fp32_cpu(x, normalized_shape, weight, bias)
        metrics1 = self.compute_metrics(ground_truth, test1_result, 
                                        "layernorm-fp32-cpu", case_name)
        self.results.append(metrics1)
        self.print_metrics(metrics1)
        
        # Test 2: fp32 on GPU (cuDNN)
        if torch.cuda.is_available():
            test2_result = self.layernorm_fp32_gpu(x, normalized_shape, weight, bias)
            metrics2 = self.compute_metrics(ground_truth, test2_result,
                                           "layernorm-fp32-gpu (cuDNN)", case_name)
            self.results.append(metrics2)
            self.print_metrics(metrics2)
        else:
            print("\n[SKIPPED] layernorm-fp32-gpu (cuDNN) - CUDA not available")
    
    def print_metrics(self, metrics: Dict):
        """Print metrics in a readable format"""
        print(f"\n{metrics['test_name']}:")
        print(f"  Max Absolute Error:  {metrics['max_abs_error']:.6e}")
        print(f"  Mean Absolute Error: {metrics['mean_abs_error']:.6e}")
        print(f"  Max Relative Error:  {metrics['max_rel_error']:.6e}")
        print(f"  Mean Relative Error: {metrics['mean_rel_error']:.6e}")
        print(f"  Inf Count:           {metrics['inf_count']}")
        print(f"  NaN Count:           {metrics['nan_count']}")
        if metrics['ground_truth_inf'] > 0 or metrics['ground_truth_nan'] > 0:
            print(f"  [WARNING] Ground truth has {metrics['ground_truth_inf']} infs "
                  f"and {metrics['ground_truth_nan']} NaNs")
    
    def run_all_tests(self):
        """Run all test cases"""
        print("\n" + "="*80)
        print("LAYERNORM FP32 PRECISION TEST")
        print("Ground Truth: layernorm-fp64-cpu (PyTorch)")
        print("Under Test: 1) layernorm-fp32-cpu (PyTorch)")
        print("            2) layernorm-fp32-gpu (cuDNN)")
        print("="*80)
        
        test_cases = self.generate_test_cases()
        
        for case_name, x, weight, bias in test_cases:
            try:
                self.run_single_test(case_name, x, weight, bias)
            except Exception as e:
                print(f"\n[ERROR] {case_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all tests"""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to summarize.")
            return
        
        # Group by test name
        test_names = list(set(r['test_name'] for r in self.results))
        
        for test_name in test_names:
            test_results = [r for r in self.results if r['test_name'] == test_name]
            
            print(f"\n{test_name}:")
            print(f"  Total test cases: {len(test_results)}")
            
            max_abs_errors = [r['max_abs_error'] for r in test_results]
            max_rel_errors = [r['max_rel_error'] for r in test_results]
            mean_abs_errors = [r['mean_abs_error'] for r in test_results]
            mean_rel_errors = [r['mean_rel_error'] for r in test_results]
            total_infs = sum(r['inf_count'] for r in test_results)
            total_nans = sum(r['nan_count'] for r in test_results)
            
            print(f"  Max absolute error across all cases: {max(max_abs_errors):.6e}")
            print(f"  Min absolute error across all cases: {min(max_abs_errors):.6e}")
            print(f"  Avg absolute error across all cases: {np.mean(max_abs_errors):.6e}")
            print(f"  Median absolute error: {np.median(max_abs_errors):.6e}")
            print(f"  Std absolute error: {np.std(max_abs_errors):.6e}")
            print(f"  95th percentile absolute error: {np.percentile(max_abs_errors, 95):.6e}")
            print(f"  Max relative error across all cases: {max(max_rel_errors):.6e}")
            print(f"  Avg relative error across all cases: {np.mean(max_rel_errors):.6e}")
            print(f"  Median relative error: {np.median(max_rel_errors):.6e}")
            print(f"  Avg of mean absolute errors: {np.mean(mean_abs_errors):.6e}")
            print(f"  Avg of mean relative errors: {np.mean(mean_rel_errors):.6e}")
            print(f"  Total Inf count: {total_infs}")
            print(f"  Total NaN count: {total_nans}")
            
            # Find worst cases
            worst_abs_idx = np.argmax(max_abs_errors)
            worst_rel_idx = np.argmax(max_rel_errors)
            
            print(f"\n  Worst absolute error case: {test_results[worst_abs_idx]['case_name']}")
            print(f"    Error: {max_abs_errors[worst_abs_idx]:.6e}")
            print(f"  Worst relative error case: {test_results[worst_rel_idx]['case_name']}")
            print(f"    Error: {max_rel_errors[worst_rel_idx]:.6e}")
        
        # Overall comparison metrics
        self.print_overall_metrics(test_names)
        
        print("\n" + "="*80)
    
    def print_overall_metrics(self, test_names: List[str]):
        """Print overall comparison metrics across all implementations"""
        print("\n" + "="*80)
        print("OVERALL METRICS - COMPARISON")
        print("="*80)
        
        if len(test_names) < 2:
            print("\nOnly one implementation tested, no comparison available.")
            return
        
        # Get all test cases that are common across implementations
        cpu_results = [r for r in self.results if 'cpu' in r['test_name'].lower() and 'gpu' not in r['test_name'].lower()]
        gpu_results = [r for r in self.results if 'gpu' in r['test_name'].lower()]
        
        if not cpu_results:
            print("\nNo CPU results to compare.")
            return
        
        if not gpu_results:
            print("\nNo GPU results to compare.")
            return
        
        print(f"\nComparing {len(cpu_results)} test cases")
        
        # Aggregate statistics
        cpu_max_abs = [r['max_abs_error'] for r in cpu_results]
        gpu_max_abs = [r['max_abs_error'] for r in gpu_results]
        cpu_mean_abs = [r['mean_abs_error'] for r in cpu_results]
        gpu_mean_abs = [r['mean_abs_error'] for r in gpu_results]
        cpu_max_rel = [r['max_rel_error'] for r in cpu_results]
        gpu_max_rel = [r['max_rel_error'] for r in gpu_results]
        
        print("\n--- Absolute Error Comparison ---")
        print(f"CPU - Overall Max: {max(cpu_max_abs):.6e}, Overall Avg: {np.mean(cpu_max_abs):.6e}")
        print(f"GPU - Overall Max: {max(gpu_max_abs):.6e}, Overall Avg: {np.mean(gpu_max_abs):.6e}")
        
        abs_diff = np.array(gpu_max_abs) - np.array(cpu_max_abs)
        print(f"Difference (GPU - CPU):")
        print(f"  Max difference: {np.max(abs_diff):.6e}")
        print(f"  Mean difference: {np.mean(abs_diff):.6e}")
        print(f"  Median difference: {np.median(abs_diff):.6e}")
        
        better_count = np.sum(abs_diff < 0)
        worse_count = np.sum(abs_diff > 0)
        equal_count = np.sum(abs_diff == 0)
        print(f"  GPU better: {better_count}/{len(abs_diff)} cases")
        print(f"  GPU worse: {worse_count}/{len(abs_diff)} cases")
        print(f"  GPU equal: {equal_count}/{len(abs_diff)} cases")
        
        print("\n--- Relative Error Comparison ---")
        print(f"CPU - Overall Max: {max(cpu_max_rel):.6e}, Overall Avg: {np.mean(cpu_max_rel):.6e}")
        print(f"GPU - Overall Max: {max(gpu_max_rel):.6e}, Overall Avg: {np.mean(gpu_max_rel):.6e}")
        
        rel_diff = np.array(gpu_max_rel) - np.array(cpu_max_rel)
        print(f"Difference (GPU - CPU):")
        print(f"  Max difference: {np.max(rel_diff):.6e}")
        print(f"  Mean difference: {np.mean(rel_diff):.6e}")
        
        print("\n--- Mean Error Comparison ---")
        print(f"CPU - Avg of mean abs errors: {np.mean(cpu_mean_abs):.6e}")
        print(f"GPU - Avg of mean abs errors: {np.mean(gpu_mean_abs):.6e}")
        
        print("\n--- Numerical Stability ---")
        cpu_infs = sum(r['inf_count'] for r in cpu_results)
        gpu_infs = sum(r['inf_count'] for r in gpu_results)
        cpu_nans = sum(r['nan_count'] for r in cpu_results)
        gpu_nans = sum(r['nan_count'] for r in gpu_results)
        
        print(f"CPU - Total Infs: {cpu_infs}, Total NaNs: {cpu_nans}")
        print(f"GPU - Total Infs: {gpu_infs}, Total NaNs: {gpu_nans}")
        
        # Error distribution analysis
        print("\n--- Error Distribution (Max Absolute Errors) ---")
        thresholds = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        print(f"{'Threshold':<15} {'CPU Count':<15} {'GPU Count':<15}")
        for threshold in thresholds:
            cpu_count = np.sum(np.array(cpu_max_abs) < threshold)
            gpu_count = np.sum(np.array(gpu_max_abs) < threshold)
            print(f"< {threshold:<13.0e} {cpu_count:<15} {gpu_count:<15}")
        
        # Overall verdict
        print("\n--- Overall Assessment ---")
        if np.mean(gpu_max_abs) < np.mean(cpu_max_abs):
            print(f"✓ GPU implementation has LOWER average max absolute error by {(1 - np.mean(gpu_max_abs)/np.mean(cpu_max_abs))*100:.2f}%")
        elif np.mean(gpu_max_abs) > np.mean(cpu_max_abs):
            print(f"✗ GPU implementation has HIGHER average max absolute error by {(np.mean(gpu_max_abs)/np.mean(cpu_max_abs) - 1)*100:.2f}%")
        else:
            print("= GPU and CPU implementations have equal average max absolute errors")
        
        if gpu_infs + gpu_nans < cpu_infs + cpu_nans:
            print(f"✓ GPU implementation has BETTER numerical stability ({gpu_infs + gpu_nans} vs {cpu_infs + cpu_nans} anomalies)")
        elif gpu_infs + gpu_nans > cpu_infs + cpu_nans:
            print(f"✗ GPU implementation has WORSE numerical stability ({gpu_infs + gpu_nans} vs {cpu_infs + cpu_nans} anomalies)")
        else:
            print(f"= Both implementations have equal numerical stability")
        
        # Precision grade
        max_error = max(max(cpu_max_abs), max(gpu_max_abs))
        if max_error < 1e-7:
            grade = "EXCELLENT"
        elif max_error < 1e-6:
            grade = "VERY GOOD"
        elif max_error < 1e-5:
            grade = "GOOD"
        elif max_error < 1e-4:
            grade = "ACCEPTABLE"
        else:
            grade = "POOR"
        
        print(f"\nOverall Precision Grade: {grade} (max error: {max_error:.6e})")

def main():
    """Main function to run the test suite"""
    # Setup logging to file
    log_file = 'fp32_layernorm.log'
    tee_logger = TeeLogger(log_file)
    original_stdout = sys.stdout
    sys.stdout = tee_logger
    
    try:
        # Print header with timestamp
        print(f"{'='*80}")
        print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Log file: {log_file}")
        print(f"{'='*80}\n")
        
        # Ensure deterministic mode is set (redundant but explicit)
        set_deterministic_mode(42)
        
        tester = LayerNormPrecisionTester()
        tester.run_all_tests()
        
        # Print footer with timestamp
        print(f"\n{'='*80}")
        print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Results saved to: {log_file}")
        print(f"{'='*80}")
        
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee_logger.close()
        print(f"Log saved to: {log_file}")

if __name__ == "__main__":
    main()
