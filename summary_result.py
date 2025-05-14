import os
import re

def extract_parameters(filename):
    """파일명에서 숫자형 하이퍼파라미터를 추출하여 리스트로 반환"""
    numbers = re.findall(r'\d+\.\d+|\d+', filename)
    return [float(num) for num in numbers]

def get_merge_method_index(filename, merge_methods_order):
    """
    병합 방식이 정확하게 포함되어 있는 경우만 인덱스를 반환.
    길이가 긴 병합 방식부터 먼저 탐색해서 겹침 문제 해결.
    """
    sorted_methods = sorted(merge_methods_order, key=len, reverse=True)
    for method in sorted_methods:
        pattern = f"_{method}_"  # 정확한 매칭을 위해 언더스코어 포함
        if pattern in filename:
            return merge_methods_order.index(method)
    return len(merge_methods_order)

def process_log_files(logs_dir):
    # 병합 방식 우선순위 정의 (실제 순서 기준)
    merge_methods_order = [
        "task_arithmetic", "ties", "dare_ties", "dare_linear",
        "breadcrumbs", "breadcrumbs_ties", "della", "della_linear", "sce"
    ]

    # logs_dir 내의 모든 .txt 파일 필터링
    files = [f for f in os.listdir(logs_dir) if f.endswith('.txt')]

    # 정렬: (병합 방식 순서, 숫자 파라미터 순서)
    sorted_files = sorted(files, key=lambda x: (
        get_merge_method_index(x, merge_methods_order),
        extract_parameters(x)
    ))

    combined_content = ""
    i = 1
    for filename in sorted_files:
        file_path = os.path.join(logs_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            # "edit finish" 이후의 내용만 추출
            if "edit finish" in content:
                post_edit_content = content.split("edit finish", 1)[1]

                # 구분자 및 파일명 출력
                combined_content += "="*50 + f"\n{i}번째 파일\n"
                combined_content += f"File: {filename}\n"
                combined_content += post_edit_content
                combined_content += "\n" + "="*50 + "\n"
                i += 1

    # 결과 파일 저장
    combined_file_path = os.path.join(logs_dir, "combined_results.txt")
    with open(combined_file_path, 'w', encoding='utf-8') as combined_file:
        combined_file.write(combined_content)

    print(f"All results have been saved to {combined_file_path}")

if __name__ == "__main__":
    logs_directory = './logs/cikm/2_25'  # 로그 폴더 경로 지정
    process_log_files(logs_directory)
