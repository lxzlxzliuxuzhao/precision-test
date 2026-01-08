from util.case_util import *
import argparse


def test(op: str, dtype: str, device: str):
    if op == "softmax":
        from cases.softmax.case import Case, caller
    elif op == "layernorm":
        from cases.layernorm.case import Case, caller
    else:
        assert False

    case_list = load_all_cases_by_name(Case, op, dtype)
    result_list = run_all_cases(case_list, caller, dtype, device)
    save_all_results_by_name(result_list, op, dtype, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, required=True, help="operator name")
    parser.add_argument("--dtype", type=str, required=True, help="data type")
    parser.add_argument("--device", type=str, required=True, help="device type")
    args = parser.parse_args()
    test(args.op, args.dtype, args.device)
