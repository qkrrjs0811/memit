import random
from copy import deepcopy
import numpy as np

from .falcon_util import *


def preprocessing(in_file) -> str:
    line = in_file.readline()
    if not line:
        return None
    
    return line.strip()


def get_performance(result: list):
    true_count = sum(result)
    total_count = len(result)
    performance = true_count / total_count

    return performance


def write_of_log_parsing(results: list, out_file_path: str):
    print(f'# write_of_log_parsing() out_file_path : {out_file_path}')

    out_file = open_file(out_file_path, mode='w')
    for result in results:
        result = [str(ent) for ent in result]
        out_str = '\t'.join(result)
        out_file.write(f'{out_str}\n')
    out_file.close()


def write_performance_matrix(identical_num, num_edits, results: list, out_file_path: str, skip_cnt=2):
    skip_idx = skip_cnt
    extend_cnt, batch_cnt = 1, -1
    p_matrix = np.zeros((identical_num+1, identical_num+1))

    for i in range(1, len(results)+1):
        if i <= skip_idx:
            # print(f'[{i}] skip')
            batch_cnt = -1
            continue
        elif i <= skip_idx + extend_cnt:
            p = get_performance(results[i-1])
            # print(f'[{i}] extend : {results[i-1]} : {p}')
            
            batch_cnt += 1
            p_matrix[batch_cnt][extend_cnt-1] = p

            if i == skip_idx + extend_cnt:
                skip_idx = (skip_idx + extend_cnt) + skip_cnt
                extend_cnt += 1
    # print()
    # print(p_matrix)

    for i in range(identical_num):
        non_zeros = p_matrix[i, :identical_num][p_matrix[i, :identical_num] != 0]
        if len(non_zeros) > 0:
            row_mean = np.mean(non_zeros)
            p_matrix[i, identical_num] = row_mean
    
    for j in range(identical_num):
        non_zeros = p_matrix[:identical_num, j][p_matrix[:identical_num, j] != 0]
        if len(non_zeros) > 0:
            col_mean = np.mean(non_zeros)
            p_matrix[identical_num, j] = col_mean
    
    non_zeros = p_matrix[identical_num, :identical_num][p_matrix[identical_num, :identical_num] != 0]
    p_matrix[identical_num, identical_num] = np.mean(non_zeros)
    # print()
    # print(p_matrix)

    # 파일로 작성
    np.savetxt(out_file_path, p_matrix, delimiter='\t', fmt='%f')


def log_parsing(in_file_path: str, out_file_path: str, identical_num=-1):
    print(f'# log_parsing() in_file_path : {in_file_path}')
    print(f'# log_parsing() identical_num : {identical_num}')

    num_edits = 0
    prefix, target_true, target_new = '', '', ''
    predict, is_correct = '', None
    results, result = [], []
    simple_results = []
    cnt = 0

    in_file = open_file(in_file_path, mode='r')
    while 1:
        line = preprocessing(in_file)
        if line is None:
            break
        elif len(line) == 0:
            continue

        if num_edits == 0 and line.startswith('num_edits : '):
            num_edits = int(line.split(':')[1].strip())
            print(f'# log_parsing() num_edits : {num_edits}\n')
        elif line.startswith('prefixes : '):
            prefix = line.split(':')[1].strip()
            prefix = prefix[2:-2]
        elif line.startswith('target_true : '):
            target_true = line.split(':')[1].strip()
        elif line.startswith('target_new : '):
            target_new = line.split(':')[1].strip()
        elif line.startswith('[0] predict : '):
            predict = line.split(':')[1].split(',')[0].strip()
        elif 'targets_correct : ' in line:
            is_correct = line.split(':')[1].strip()[1:-1]
            result.append(prefix)
            result.append(target_true)
            result.append(target_new)
            result.append(predict)
            result.append(is_correct)
            
            results.append(deepcopy(result))
            result.clear()

            if len(results) == num_edits:
                cnt += 1
                write_of_log_parsing(results, out_file_path.format(f'_({cnt}).txt'))

                simple_result = [result[4] == 'True' for result in results]
                simple_results.append(simple_result)
                results.clear()
    in_file.close()

    write_of_log_parsing(simple_results, out_file_path.format('_(simple).txt'))
    write_json_to_file(simple_results, out_file_path.format('_(simple).json'))

    if identical_num > 0:
        write_performance_matrix(identical_num, num_edits, simple_results, out_file_path.format('_p_matrix.txt'))


            



def run():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    log_dir = f'{home_dir}/logs/241128_multiple_sr_both_r_18~22'

    file_names = ['log_test_identical4_20{}']

    for file_name in file_names:
        in_file_path = f'{log_dir}/{file_name}'
        out_file_path = f'{log_dir}/log_processing/' + file_name.format('') + f'/{file_name}'
        log_parsing(in_file_path.format('')+'.txt', out_file_path)


def run_241206_baseline_sequential():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    log_dir = f'{home_dir}/logs/241126_baseline/sequential_identical_subjects'

    in_file_paths = get_file_paths(log_dir, False)
    for in_file_path in in_file_paths:
        file_name = get_file_name(in_file_path, True)
        idx = file_name.find('identical')
        identical_num = file_name[idx+9]

        # if 'log_mcf_sequential_identical4_subjects_all_batch5' == file_name:
        if 'log_mcf_sequential_identical2_subjects_all_batch100' != file_name:
            out_file_path = f'{log_dir}/log_processing/{file_name}/{file_name}' + '{}'
            log_parsing(in_file_path, out_file_path, int(identical_num))





if __name__ == "__main__":
    # run()
    run_241206_baseline_sequential()

