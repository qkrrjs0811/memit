import torch
import yaml
import argparse
import sys
from pathlib import Path

# mergekit이 설치된 경로 추가 (import 오류 방지)
sys.path.append(str(Path(__file__).resolve().parent / "mergekit_local"))  # 필요시 수정

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# Merge 옵션
LORA_MERGE_CACHE = "/tmp"
COPY_TOKENIZER = True
LAZY_UNPICKLE = False
LOW_CPU_MEMORY = False

# 각 방법론에 필요한 파라미터 정의
merge_method_params = {
    "task_arithmetic": ["lambda"],
    "ties": ["lambda", "density"],
    "dare_ties": ["lambda", "density"],
    "dare_linear": ["lambda"],
    "breadcrumbs": ["lambda", "density", "gamma"],
    "breadcrumbs_ties": ["lambda", "density", "gamma"],
    "della": ["lambda", "density", "epsilon"],
    "della_linear": ["lambda", "density", "epsilon"],
    "sce": ["lambda", "density", "select_topk"]
}

def create_merge_conf(base_model, merge_method, params, model_dirs):
    config = {
        "base_model": base_model,
        "dtype": "float16",
        "merge_method": merge_method,
        "models": [{"model": model_dir, "parameters": {"weight": 1.0}} for model_dir in model_dirs],
        "parameters": params
    }
    with open("merge_conf.yml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)

def main():
    parser = argparse.ArgumentParser(description='Merge models with different methods and hyperparameters.')
    parser.add_argument('--model_dirs', type=str, nargs='+', required=False, help='List of model directories to merge', default=None)
    parser.add_argument('--merge_methods', type=str, nargs='+', required=True, help='List of merge methods')
    parser.add_argument('--lambdas', type=float, nargs='+', required=False, help='List of lambda values', default=[0.1])
    parser.add_argument('--densities', type=float, nargs='+', required=False, help='List of density values', default=[0.1])
    parser.add_argument('--epsilons', type=float, nargs='+', required=False, help='List of epsilon values', default=[0.1])
    parser.add_argument('--gammas', type=float, nargs='+', required=False, help='List of gamma values', default=[0.1])
    parser.add_argument('--select_topk', type=int, nargs='+', required=False, help='List of select_topk values', default=[1])

    args = parser.parse_args()

    # List all directories in the './edited_models' path
    all_model_dirs = [str(p) for p in Path('./edited_models').iterdir() if p.is_dir()]

    # Add gpt2-xl path if it's in args.model_dirs
    if 'gpt2-xl' in args.model_dirs:
        all_model_dirs.append('original_models/gpt2-xl')

    # Filter model directories based on user input
    if args.model_dirs:
        model_dirs = [d for d in all_model_dirs if Path(d).name in args.model_dirs]
    else:
        model_dirs = all_model_dirs




    # 각 방법론에 대한 파라미터 조합 생성
    for merge_method in args.merge_methods:
        params_list = merge_method_params[merge_method]
        
        # Initialize the parameter combinations
        param_combinations = [{}]
        
        # Iterate over each parameter type and update combinations
        if "lambda" in params_list:
            param_combinations = [
                {**comb, "lambda": lambda_val} for comb in param_combinations for lambda_val in args.lambdas
            ]
        if "density" in params_list:
            param_combinations = [
                {**comb, "density": density} for comb in param_combinations for density in args.densities
            ]
        if "epsilon" in params_list:
            param_combinations = [
                {**comb, "epsilon": epsilon} for comb in param_combinations for epsilon in args.epsilons
            ]
        if "gamma" in params_list:
            param_combinations = [
                {**comb, "gamma": gamma} for comb in param_combinations for gamma in args.gammas
            ]
        if "select_topk" in params_list:
            param_combinations = [
                {**comb, "select_topk": select_topk} for comb in param_combinations for select_topk in args.select_topk
            ]

        # Iterate over the generated parameter combinations
        for params in param_combinations:
            # Check for valid parameter combinations
            if merge_method in ['della', 'della_linear'] and ("density" in params and "epsilon" in params) and (params["density"] + params["epsilon"] >= 1 or params["density"] - params["epsilon"] <= 0):
                print(f"Error: Invalid combination for {merge_method} with density {params['density']} and epsilon {params['epsilon']}")
                continue

            create_merge_conf("gpt2-xl", merge_method, params, model_dirs)
            print("########################################################")
            for model_dir in model_dirs:
                # Extract the parts from the directory name
                parts = Path(model_dir).name.split('_')
                if len(parts) == 4:
                    identical_num, num_edits, batch_idx = parts[1], parts[2], parts[3]
                    if 'gpt2-xl' in model_dirs[1]:
                        OUTPUT_PATH = f"./merged/merged_gpt2_xl_{identical_num}_{num_edits}_{batch_idx}_{merge_method}_{params}"
                    else:
                        OUTPUT_PATH = f"./merged/merged_{identical_num}_{num_edits}_{merge_method}_{params}"
                    if Path(OUTPUT_PATH).exists():
                        print(f"\n[Warning] 병합된 모델이 이미 존재합니다: {OUTPUT_PATH}")
                        print("삭제 후 다시 실행하거나 다른 이름을 입력하세요.")
                        continue
                    print(f"\n[MergeRunner] merge_method: {merge_method}, params: {params}")
                    print(f"[MergeRunner] 병합 결과 경로: {OUTPUT_PATH}")
                    with open("merge_conf.yml", "r", encoding="utf-8") as f:
                        merge_config = MergeConfiguration.model_validate(yaml.safe_load(f))
                    run_merge(
                        merge_config,
                        out_path=OUTPUT_PATH,
                        options=MergeOptions(
                            lora_merge_cache=LORA_MERGE_CACHE,
                            cuda=torch.cuda.is_available(),
                            copy_tokenizer=COPY_TOKENIZER,
                            lazy_unpickle=LAZY_UNPICKLE,
                            low_cpu_memory=LOW_CPU_MEMORY,
                        ),
                    )
                    print(f"\n[MergeRunner] 병합 완료! 모델이 {OUTPUT_PATH}에 저장되었습니다.")
                    break

if __name__ == "__main__":
    main()
