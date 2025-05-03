import json
import random
import argparse
from pathlib import Path

def sample_json(input_path, output_path, sample_size, seed=42):
    # 데이터 로드
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📦 Loaded {len(data)} samples from {input_path}")

    # 시드 고정 후 샘플링
    random.seed(seed)
    sampled_data = random.sample(data, sample_size)

    # 결과 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(sampled_data)} samples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to the original JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save the sampled JSON")
    parser.add_argument("--size", type=int, required=True, help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default=42)")
    args = parser.parse_args()

    sample_json(args.input, args.output, args.size, args.seed)
