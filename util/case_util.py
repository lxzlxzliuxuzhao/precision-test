import os
import torch


def save_case(case, path: str):
    case.save(path)

def save_all_cases(case_list: list, path: str):
    assert os.path.exists(path)
    path_cases = os.path.join(path, "cases")
    os.makedirs(path_cases, exist_ok=True)
    for i, case in enumerate(case_list):
        save_case(case, os.path.join(path_cases, f"{i}.pt"))

def save_all_cases_by_name(case_list: dict, op: str, dtype: str):
    path = os.path.join("cases", op, dtype)
    save_all_cases(case_list, path)
    

def load_case(CaseClass: type, path: str):
    assert os.path.exists(path)
    case = CaseClass()
    case.load(path)
    return case

def load_all_cases(CaseClass: type, path: str) -> list:
    assert os.path.exists(path)
    case_list = []
    path_cases = os.path.join(path, "cases")
    n_cases = len(os.listdir(path_cases))
    for i in range(n_cases):
        case = load_case(CaseClass, os.path.join(path_cases, f"{i}.pt"))
        case_list.append(case)
    return case_list

def load_all_cases_by_name(CaseClass: type, op: str, dtype: str) -> list:
    path = os.path.join("cases", op, dtype)
    return load_all_cases(CaseClass, path)


def save_result(result, path: str):
    result.save(path)

def save_all_results(result_list: list, path: str, device: str):
    assert os.path.exists(path)
    path_results_device = os.path.join(path, "results", device)
    os.makedirs(path_results_device, exist_ok=True)
    for i, result in enumerate(result_list):
        save_result(result, os.path.join(path_results_device, f"{i}.pt"))

def save_all_results_by_name(result_list: list, op: str, dtype: str, device: str):
    path = os.path.join("cases", op, dtype)
    save_all_results(result_list, path, device)


def load_result(ResultClass: type, path: str):
    assert os.path.exists(path)
    result = ResultClass()
    result.load(path)
    return result

def load_all_results(ResultClass: type, path: str, device: str) -> list:
    assert os.path.exists(path)
    result_list = []
    path_results_device = os.path.join(path, "results", device)
    n_results = len(os.listdir(path_results_device))
    for i in range(n_results):
        result = load_result(ResultClass, os.path.join(path_results_device, f"{i}.pt"))
        result_list.append(result)
    return result_list

def load_all_results_by_name(ResultClass: type, op: str, dtype: str, device: str) -> list:
    path = os.path.join("cases", op, dtype)
    return load_all_results(ResultClass, path, device)


def run_case(case, caller, dtype: str, device: str):
    return caller(case, dtype=dtype, device=device)

def run_all_cases(case_list: list, caller, dtype: str, device: str) -> list:
    result_list = []
    for case in case_list:
        result = run_case(case, caller, dtype=dtype, device=device)
        result_list.append(result)
    return result_list


def get_torch_dtype(dtype: str):
    if dtype == "fp32":
        return torch.float32
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "fp64":
        return torch.float64
    else:
        assert False

def get_torch_device(device: str):
    if device == "cpu" or device == "cuda" or device == "npu":
        return device
    elif device == "baseline":
        return "cpu"
    elif device == "iluvatar":
        return "cuda"
    else:   
        assert False
