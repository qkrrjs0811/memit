import re

def extract_high_performance_methods(file_path, threshold):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 각 파일 블록 기준으로 분할
    blocks = content.split("==================================================")

    results = []

    for block in blocks:
        # 파일명 추출
        file_match = re.search(r"File:\s*(.+\.txt)", block)
        edited_avg_match = re.search(r"edited \(avg\)\s*:\s*\[?([0-9.]+)", block)

        if file_match and edited_avg_match:
            file_name = file_match.group(1).strip()
            edited_avg = float(edited_avg_match.group(1))

            if edited_avg >= threshold:
                results.append((file_name, edited_avg))

    return results


# 사용 예시
high_perf_files = extract_high_performance_methods("logs/cikm/2_25/combined_results.txt", 0.82)

# 결과를 파일로 저장
output_file_path = "logs/cikm/2_25/high_performance_methods.txt"
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for file, score in high_perf_files:
        output_file.write(f"{file} - edited avg: {score}\n")

# 결과를 출력
for file, score in high_perf_files:
    print(f"{file} - edited avg: {score}")
