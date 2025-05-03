import torch
import yaml
import argparse
import sys
from pathlib import Path

# mergekit이 설치된 경로 추가 (import 오류 방지)
sys.path.append(str(Path(__file__).resolve().parent / "mergekit_local"))  # 필요시 수정

from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

# YAML 병합 구성 파일 경로
CONFIG_YML = "./merge_conf.yml"

# Merge 옵션
LORA_MERGE_CACHE = "/tmp"
COPY_TOKENIZER = True
LAZY_UNPICKLE = False
LOW_CPU_MEMORY = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_name", type=str, required=True, help="병합된 모델의 이름")
    args = parser.parse_args()

    OUTPUT_PATH = f"./merged/{args.output_model_name}"

    if Path(OUTPUT_PATH).exists():
        print(f"[Warning] 병합된 모델이 이미 존재합니다: {OUTPUT_PATH}")
        print("삭제 후 다시 실행하거나 다른 이름을 입력하세요.")
        return

    print(f"[MergeRunner] 병합 구성 파일: {CONFIG_YML}")
    print(f"[MergeRunner] 병합 결과 경로: {OUTPUT_PATH}")

    with open(CONFIG_YML, "r", encoding="utf-8") as f:
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

    print(f"[MergeRunner] 병합 완료! 모델이 {OUTPUT_PATH}에 저장되었습니다.")

if __name__ == "__main__":
    main()
