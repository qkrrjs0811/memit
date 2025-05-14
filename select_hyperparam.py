import re
from collections import Counter

def extract_hyperparameters(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lambda_counter = Counter()
    density_counter = Counter()
    epsilon_counter = Counter()

    for line in lines:
        lambda_match = re.search(r"lambda': ([0-9.]+)", line)
        density_match = re.search(r"density': ([0-9.]+)", line)
        epsilon_match = re.search(r"epsilon': ([0-9.]+)", line)

        if lambda_match:
            lambda_counter[float(lambda_match.group(1))] += 1
        if density_match:
            density_counter[float(density_match.group(1))] += 1
        if epsilon_match:
            epsilon_counter[float(epsilon_match.group(1))] += 1

    return lambda_counter, density_counter, epsilon_counter

def analyze_hyperparameters(high_perf_file, combined_file):
    # high_performance_methods.txt에서 성공한 경우의 하이퍼파라미터 카운트
    success_lambda, success_density, success_epsilon = extract_hyperparameters(high_perf_file)
    
    # combined_results.txt에서 전체 하이퍼파라미터 카운트
    total_lambda, total_density, total_epsilon = extract_hyperparameters(combined_file)
    
    # 성공 비율 계산
    lambda_ratios = {k: (success_lambda[k], total_lambda[k]) for k in total_lambda.keys()}
    density_ratios = {k: (success_density[k], total_density[k]) for k in total_density.keys()}
    epsilon_ratios = {k: (success_epsilon[k], total_epsilon[k]) for k in total_epsilon.keys()}
    
    return lambda_ratios, density_ratios, epsilon_ratios

# 사용 예시
lambda_ratios, density_ratios, epsilon_ratios = analyze_hyperparameters(
    "logs/cikm/2_25/high_performance_methods.txt",
    "logs/cikm/2_25/combined_results.txt"
)

print("\nLambda 성공 비율:")
for lambda_val, (success, total) in sorted(lambda_ratios.items()):
    ratio = success / total
    print(f"lambda={lambda_val}: {ratio:.2%} ({success}/{total})")

print("\nDensity 성공 비율:")
for density_val, (success, total) in sorted(density_ratios.items()):
    ratio = success / total
    print(f"density={density_val}: {ratio:.2%} ({success}/{total})")

print("\nEpsilon 성공 비율:")
for epsilon_val, (success, total) in sorted(epsilon_ratios.items()):
    ratio = success / total
    print(f"epsilon={epsilon_val}: {ratio:.2%} ({success}/{total})")
