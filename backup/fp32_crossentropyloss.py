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

class CrossEntropyLossPrecisionTester:
    """Test CrossEntropyLoss precision with fp32 vs fp64 ground truth"""
    
    def __init__(self):
        self.results = []
        
    def compute_metrics(self, ground_truth: torch.Tensor, test_result: torch.Tensor, 
                       test_name: str, case_name: str) -> Dict:
        """Compute precision metrics between ground truth and test result"""
        
        # Convert to numpy for easier computation (detach first to handle requires_grad)
        gt = ground_truth.detach().cpu().numpy().astype(np.float64)
        test = test_result.detach().cpu().numpy().astype(np.float64)
        
        # For scalar loss values, treat as arrays
        if gt.shape == ():
            gt = np.array([gt])
            test = np.array([test])
        
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
    
    def cross_entropy_fp64_cpu(self, logits: torch.Tensor, targets: torch.Tensor, 
                                reduction: str = 'mean') -> torch.Tensor:
        """Ground truth: CrossEntropyLoss with fp64 on CPU"""
        logits_fp64 = logits.to(dtype=torch.float64, device='cpu')
        targets_fp64 = targets.to(device='cpu')
        if targets.dtype == torch.float32 or targets.dtype == torch.float64:
            targets_fp64 = targets_fp64.to(dtype=torch.float64)
        
        # Handle 3D logits (batch, seq, vocab) -> reshape to (batch*seq, vocab)
        original_shape = logits_fp64.shape
        if logits_fp64.dim() == 3:
            batch_size, seq_len, vocab_size = logits_fp64.shape
            logits_fp64 = logits_fp64.view(-1, vocab_size)
            targets_fp64 = targets_fp64.view(-1)
        
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        loss = criterion(logits_fp64, targets_fp64)
        
        # If reduction is 'none' and we reshaped, reshape back
        if reduction == 'none' and len(original_shape) == 3:
            loss = loss.view(original_shape[0], original_shape[1])
        
        return loss
    
    def cross_entropy_fp32_cpu(self, logits: torch.Tensor, targets: torch.Tensor,
                                reduction: str = 'mean') -> torch.Tensor:
        """Test 1: CrossEntropyLoss with fp32 on CPU"""
        logits_fp32 = logits.to(dtype=torch.float32, device='cpu')
        targets_fp32 = targets.to(device='cpu')
        if targets.dtype == torch.float32 or targets.dtype == torch.float64:
            targets_fp32 = targets_fp32.to(dtype=torch.float32)
        
        # Handle 3D logits (batch, seq, vocab) -> reshape to (batch*seq, vocab)
        original_shape = logits_fp32.shape
        if logits_fp32.dim() == 3:
            batch_size, seq_len, vocab_size = logits_fp32.shape
            logits_fp32 = logits_fp32.view(-1, vocab_size)
            targets_fp32 = targets_fp32.view(-1)
        
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        loss = criterion(logits_fp32, targets_fp32)
        
        # If reduction is 'none' and we reshaped, reshape back
        if reduction == 'none' and len(original_shape) == 3:
            loss = loss.view(original_shape[0], original_shape[1])
        
        return loss
    
    def cross_entropy_fp32_gpu(self, logits: torch.Tensor, targets: torch.Tensor,
                                reduction: str = 'mean') -> torch.Tensor:
        """Test 2: CrossEntropyLoss with fp32 on GPU"""
        if not torch.cuda.is_available():
            return None
        logits_fp32 = logits.to(dtype=torch.float32, device='cuda')
        targets_fp32 = targets.to(device='cuda')
        if targets.dtype == torch.float32 or targets.dtype == torch.float64:
            targets_fp32 = targets_fp32.to(dtype=torch.float32)
        
        # Handle 3D logits (batch, seq, vocab) -> reshape to (batch*seq, vocab)
        original_shape = logits_fp32.shape
        if logits_fp32.dim() == 3:
            batch_size, seq_len, vocab_size = logits_fp32.shape
            logits_fp32 = logits_fp32.view(-1, vocab_size)
            targets_fp32 = targets_fp32.view(-1)
        
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        loss = criterion(logits_fp32, targets_fp32)
        
        # If reduction is 'none' and we reshaped, reshape back
        if reduction == 'none' and len(original_shape) == 3:
            loss = loss.view(original_shape[0], original_shape[1])
        
        return loss
    
    def generate_test_cases(self) -> List[Tuple[str, torch.Tensor, torch.Tensor, str]]:
        """Generate comprehensive test cases for LLM training scenarios
        Returns: List of (case_name, logits, targets, reduction)
        """
        test_cases = []
        batch_size = 16
        seq_len = 128
        vocab_size = 50000  # Typical LLM vocab size
        
        # Case 1: Normal distribution logits (typical case)
        logits = torch.randn(batch_size, vocab_size) * 2
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Normal logits N(0,2)",
            logits, targets, 'mean'
        ))
        
        # Case 2: Large positive logits (overflow risk)
        logits = torch.rand(batch_size, vocab_size) * 50 + 50
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Large positive logits [50,100]",
            logits, targets, 'mean'
        ))
        
        # Case 3: Large negative logits (underflow risk)
        logits = torch.rand(batch_size, vocab_size) * (-50) - 50
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Large negative logits [-100,-50]",
            logits, targets, 'mean'
        ))
        
        # Case 4: Mixed extreme logits
        logits = torch.randn(batch_size, vocab_size) * 30
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Mixed extreme logits",
            logits, targets, 'mean'
        ))
        
        # Case 5: One dominant logit (confident prediction)
        logits = torch.randn(batch_size, vocab_size) * 0.1
        for i in range(batch_size):
            dominant_idx = torch.randint(0, vocab_size, (1,)).item()
            logits[i, dominant_idx] = 50.0
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "One dominant logit per sample",
            logits, targets, 'mean'
        ))
        
        # Case 6: All equal logits (maximum uncertainty)
        logits = torch.ones(batch_size, vocab_size) * 5.0
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "All equal logits",
            logits, targets, 'mean'
        ))
        
        # Case 7: Near-zero logits
        logits = torch.randn(batch_size, vocab_size) * 1e-3
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Near-zero logits [-1e-3,1e-3]",
            logits, targets, 'mean'
        ))
        
        # Case 8: Sequence prediction (3D logits)
        logits = torch.randn(batch_size, seq_len, vocab_size) * 5
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        test_cases.append((
            "Sequence prediction [batch, seq, vocab]",
            logits, targets, 'mean'
        ))
        
        # Case 9: Large vocabulary (stress test)
        large_vocab = 100000
        logits = torch.randn(8, large_vocab) * 3
        targets = torch.randint(0, large_vocab, (8,))
        test_cases.append((
            "Large vocabulary [100k]",
            logits, targets, 'mean'
        ))
        
        # Case 10: Small vocabulary
        small_vocab = 100
        logits = torch.randn(batch_size, small_vocab) * 3
        targets = torch.randint(0, small_vocab, (batch_size,))
        test_cases.append((
            "Small vocabulary [100]",
            logits, targets, 'mean'
        ))
        
        # Case 11: Near fp32 precision limit
        logits = torch.ones(batch_size, vocab_size) * 16777216.0 + torch.randn(batch_size, vocab_size)
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Near fp32 precision limit",
            logits, targets, 'mean'
        ))
        
        # Case 12: Correct class has very negative logit (worst case)
        logits = torch.randn(batch_size, vocab_size) * 2
        targets = torch.randint(0, vocab_size, (batch_size,))
        for i in range(batch_size):
            logits[i, targets[i]] = -50.0
        test_cases.append((
            "Correct class very negative",
            logits, targets, 'mean'
        ))
        
        # Case 13: Correct class has very positive logit (best case)
        logits = torch.randn(batch_size, vocab_size) * 2
        targets = torch.randint(0, vocab_size, (batch_size,))
        for i in range(batch_size):
            logits[i, targets[i]] = 50.0
        test_cases.append((
            "Correct class very positive",
            logits, targets, 'mean'
        ))
        
        # Case 14: Gradient-like small values
        logits = torch.rand(batch_size, vocab_size) * 0.01 + 1e-4
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Gradient-like small logits",
            logits, targets, 'mean'
        ))
        
        # Case 15: Sum reduction
        logits = torch.randn(batch_size, vocab_size) * 5
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Sum reduction",
            logits, targets, 'sum'
        ))
        
        # Case 16: None reduction (element-wise)
        logits = torch.randn(batch_size, vocab_size) * 5
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "None reduction (element-wise)",
            logits, targets, 'none'
        ))
        
        # Case 17: Alternating high/low confidence
        logits = torch.randn(batch_size, vocab_size) * 0.5
        targets = torch.randint(0, vocab_size, (batch_size,))
        for i in range(0, batch_size, 2):
            logits[i, targets[i]] = 30.0  # High confidence
        for i in range(1, batch_size, 2):
            logits[i, targets[i]] = -30.0  # Low confidence
        test_cases.append((
            "Alternating high/low confidence",
            logits, targets, 'mean'
        ))
        
        # Case 18: Exponential distribution
        logits = torch.randn(batch_size, vocab_size).abs() * 10
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Exponential distribution",
            logits, targets, 'mean'
        ))
        
        # Case 19: Very large sequence (memory stress)
        long_seq = 2048
        logits = torch.randn(4, long_seq, vocab_size) * 3
        targets = torch.randint(0, vocab_size, (4, long_seq))
        test_cases.append((
            "Long sequence [2048]",
            logits, targets, 'mean'
        ))
        
        # Case 20: Single sample
        logits = torch.randn(1, vocab_size) * 5
        targets = torch.randint(0, vocab_size, (1,))
        test_cases.append((
            "Single sample",
            logits, targets, 'mean'
        ))
        
        # Case 21: Sparse high logits
        logits = torch.ones(batch_size, vocab_size) * (-10)
        for i in range(batch_size):
            high_indices = torch.randint(0, vocab_size, (5,))
            logits[i, high_indices] = 20.0
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Sparse high logits",
            logits, targets, 'mean'
        ))
        
        # Case 22: All targets same class
        logits = torch.randn(batch_size, vocab_size) * 5
        targets = torch.ones(batch_size, dtype=torch.long) * (vocab_size // 2)
        test_cases.append((
            "All targets same class",
            logits, targets, 'mean'
        ))
        
        # Case 23: Logits near overflow boundary
        logits = torch.rand(batch_size, vocab_size) * 10 + 80
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Near overflow [80,90]",
            logits, targets, 'mean'
        ))
        
        # Case 24: Small differences in logits (precision test)
        base_logit = 10.0
        logits = torch.ones(batch_size, vocab_size) * base_logit
        logits += torch.randn(batch_size, vocab_size) * 1e-6
        targets = torch.randint(0, vocab_size, (batch_size,))
        test_cases.append((
            "Small logit differences (1e-6)",
            logits, targets, 'mean'
        ))
        
        # Case 25: Large batch size
        large_batch = 512
        logits = torch.randn(large_batch, vocab_size) * 3
        targets = torch.randint(0, vocab_size, (large_batch,))
        test_cases.append((
            "Large batch [512]",
            logits, targets, 'mean'
        ))
        
        return test_cases
    
    def run_single_test(self, case_name: str, logits: torch.Tensor, 
                       targets: torch.Tensor, reduction: str):
        """Run a single test case"""
        print(f"\n{'='*80}")
        print(f"Test Case: {case_name}")
        print(f"Logits shape: {logits.shape}, Range: [{logits.min().item():.4e}, {logits.max().item():.4e}]")
        print(f"Targets shape: {targets.shape}, Unique classes: {len(torch.unique(targets))}")
        print(f"Reduction: {reduction}")
        print(f"{'='*80}")
        
        # Ground truth: fp64 on CPU
        ground_truth = self.cross_entropy_fp64_cpu(logits, targets, reduction)
        print(f"Ground truth loss (fp64): {ground_truth.item() if ground_truth.numel() == 1 else ground_truth.mean().item():.6e}")
        
        # Test 1: fp32 on CPU
        test1_result = self.cross_entropy_fp32_cpu(logits, targets, reduction)
        metrics1 = self.compute_metrics(ground_truth, test1_result, 
                                        "crossentropy-fp32-cpu", case_name)
        self.results.append(metrics1)
        self.print_metrics(metrics1)
        
        # Clean up CPU tensors
        del test1_result
        
        # Test 2: fp32 on GPU
        if torch.cuda.is_available():
            try:
                test2_result = self.cross_entropy_fp32_gpu(logits, targets, reduction)
                metrics2 = self.compute_metrics(ground_truth, test2_result,
                                               "crossentropy-fp32-gpu", case_name)
                self.results.append(metrics2)
                self.print_metrics(metrics2)
                
                # Clean up GPU tensors and cache
                del test2_result
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\n[SKIPPED] crossentropy-fp32-gpu - Out of memory: {str(e)}")
                    torch.cuda.empty_cache()
                else:
                    raise
        else:
            print("\n[SKIPPED] crossentropy-fp32-gpu - CUDA not available")
        
        # Clean up ground truth
        del ground_truth
    
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
        print("CROSS ENTROPY LOSS FP32 PRECISION TEST")
        print("Ground Truth: crossentropy-fp64-cpu (PyTorch)")
        print("Under Test: 1) crossentropy-fp32-cpu (PyTorch)")
        print("            2) crossentropy-fp32-gpu (cuDNN)")
        print("="*80)
        
        test_cases = self.generate_test_cases()
        
        for case_name, logits, targets, reduction in test_cases:
            try:
                self.run_single_test(case_name, logits, targets, reduction)
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
    log_file = 'fp32_crossentropyloss.log'
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
        
        tester = CrossEntropyLossPrecisionTester()
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
