import random
from copy import deepcopy

from .falcon_util import *
from .model_editor import *


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def random_sampling(datas, n=100, seed=SEED):
    random.seed(seed)
    sampled = random.sample(datas, n)
    print(f'# data_preprocessor.random_sampling() data size : {len(datas)}, sampled : {len(sampled)}')

    return sampled


def get_subject_groups(datas, do_print=True):
    subject_groups = {}

    for data in datas:
        subject = data['requested_rewrite']['subject'] # 대/소문자 구분 안 해도 됨

        if not subject in subject_groups.keys():
            subject_groups[subject] = []
        subject_groups[subject].append(data)

    if do_print:
        data_size = len(datas)
        key_size = len(subject_groups.keys())
        value_size = sum(len(value) for value in subject_groups.values())

        subject_lower_set = set()
        for subject in subject_groups.keys():
            subject_lower_set.add(subject.lower())
        key_lower_size = len(subject_lower_set)

        print(f'\n# data_preprocessor.get_subject_groups() datas size : {data_size}')
        print(f'# data_preprocessor.get_subject_groups() subject_groups key size : {key_size}, lower size : {key_lower_size}')
        print(f'# data_preprocessor.get_subject_groups() subject_groups value size : {value_size}\n')

    return subject_groups


def get_subject_groups_list(datas, do_print=True):
    subject_groups = get_subject_groups(datas, do_print)

    group_sizes = {}
    for subject in subject_groups.keys():
        subject_group = subject_groups[subject]
        group_size = len(subject_group)
        add_dict_freq(group_sizes, group_size)

    subject_groups_list = [[] for _ in range(len(group_sizes))]

    for subject in subject_groups.keys():
        subject_group = subject_groups[subject]
        group_size = len(subject_group)
        subject_groups_list[group_size-1].extend(subject_group)
    
    if do_print:
        print(f'# data_preprocessor.get_subject_groups_list() print\n\tgroup sizes : {group_sizes}\n')
        for i, subject_groups in enumerate(subject_groups_list):
            print(f'\tgroup size : {i+1}, group value size : {len(subject_groups)}')
        print()

    return subject_groups_list


def replace_simple(datas):
    replace_datas = []
    for data in datas:
        case_id = data['case_id']
        subject = data['requested_rewrite']['subject']
        prompt = data['requested_rewrite']['prompt']

        # replace_data = {'case_id': case_id, 'subject': subject, 'prompt': prompt}
        replace_data = {'case_id': case_id, 'requested_rewrite': {'subject': subject, 'prompt': prompt}}

        replace_datas.append(replace_data)
    
    return replace_datas


def load_datas(in_file_path: str):
    in_file = open_file(in_file_path, mode='r')
    datas = json.load(in_file)

    print(f'# data_preprocessor.load_datas() datas size : {len(datas)}, in_file_path : {in_file_path}')
    return datas


def write_datas(out_file_path: str, datas, ext_n=-1, ext_rn=-1, do_simple=False):
    if do_simple:
        datas = replace_simple(datas)

    if ext_n > 0 and ext_n <= len(datas):
        out_file_path = out_file_path.format(f'ext_n_{ext_n}')
        datas = datas[:ext_n]
    elif ext_rn > 0 and ext_rn <= len(datas):
        out_file_path = out_file_path.format(f'ext_rn_{ext_rn}')
        datas = random_sampling(datas, ext_rn)
    else:
        out_file_path = out_file_path.format(f'all_{len(datas)}')

    make_parent(out_file_path)
    out_file = open_file(out_file_path, mode='w')
    out_file.write(to_json_str(datas))
    out_file.close()
    print(f'# data_preprocessor.write_datas() data size : {len(datas)} -> {out_file_path}')


def make_datas_multiple(datas1, datas2, out_file_path: str, step=10):
    len1, len2 = len(datas1), len(datas2)
    print(f'\n# data_preprocessor.make_datas_multiple() len1 : {len1}, len2 : {len2}')

    if len1 == len2:
        write_datas(out_file_path.format(f'{step}:0'), datas1)
        write_datas(out_file_path.format(f'0:{step}'), datas2)

        step_size = int(len1 / step)

        for i in range(1, step):
            idx = i * step_size

            datas = []
            datas.extend(datas1[:-idx])
            datas.extend(datas2[:idx])

            write_datas(out_file_path.format(f'{step-i}:{i}'), datas)


def make_datas_sequential(datas_list, out_file_path: str):
    group_size_max = len(datas_list) + 1
    merged_datas_batchs = [[] for _ in range(group_size_max)]

    for i, datas in enumerate(datas_list):
        group_size = i+2
        datas_batchs = make_datas_sequential_each(group_size, datas, out_file_path)

        for i, datas_batch in enumerate(datas_batchs):
            merged_datas_batchs[i].extend(datas_batch)
    
    merged_datas_all = []
    for i, merged_datas_batch in enumerate(merged_datas_batchs):
        merged_datas_all.extend(merged_datas_batch)
        write_datas(out_file_path.format(f'merged', '', f'merged_batch{i+1}'), merged_datas_batch)
    write_datas(out_file_path.format(f'merged', '', f'merged_all'), merged_datas_all)


def make_datas_sequential_each(group_size: int, datas, out_file_path: str):
    datas_batchs = [[] for _ in range(group_size)]

    for i, data in enumerate(datas):
        datas_batchs[i%group_size].append(data)
    
    datas_all = []
    for i, datas_batch in enumerate(datas_batchs):
        datas_all.extend(datas_batch)
        _out_file_path = out_file_path.format(f'each/identical{group_size}', group_size, f'batch{i+1}')

        if check_identical(datas_batch):
            write_datas(_out_file_path, datas_batch)
        else:
            print('### error : ' + _out_file_path + ' is not identical!')
    
    _out_file_path = out_file_path.format(f'each/identical{group_size}', group_size, 'all')
    write_datas(_out_file_path, datas_all)
    
    return datas_batchs


def check_identical(datas):
    data_size = len(datas)

    subject_set = set()
    for data in datas:
        subject = data['requested_rewrite']['subject']
        subject_set.add(subject)
    
    subject_size = len(subject_set)

    if data_size == subject_size:
        return True

    return False




def get_model_editor(num_edits=100):
    alg_name = 'MEMIT'
    model_name = 'gpt2-xl'
    hparams_fname = f'{model_name}.json'
    ds_name = 'mcf'
    # num_edits = 100

    dataset_size_limit = None
    continue_from_run = None
    skip_generation_tests = False
    generation_test_interval = 1
    conserve_memory = False
    dir_name = alg_name
    use_cache = False

    model_editor = ModelEditor(
        alg_name, model_name, hparams_fname, ds_name,
        dataset_size_limit, continue_from_run, skip_generation_tests,
        generation_test_interval, conserve_memory, dir_name, num_edits, use_cache
    )

    return model_editor





def run(out_path, datas=None):
    if datas is None:
        model_editor = get_model_editor()
        model_editor.load_data()
        datas = model_editor._ds


    '''
        [
            [data1: dict, data2: dict, ...],
            [data1: dict, data2: dict, ...],
            ...
        ]
    '''
    subject_groups_list = get_subject_groups_list(datas, True)
    ext_n = 1000

    for i in range(len(subject_groups_list)):
        out_file_path = out_path + f'/multi_counterfact_identical{i+1}' + '_{}.json'
        write_datas(out_file_path, subject_groups_list[i])

        if i == 0:
            write_datas(out_file_path, subject_groups_list[i], ext_rn=ext_n)
        elif i == 1:
            write_datas(out_file_path, subject_groups_list[i], ext_n=ext_n)


    # 1. Multiple Identical Subjects datasets
    in_file_path1 = out_path + f'/multi_counterfact_identical1_ext_rn_{ext_n}.json'
    in_file_path2 = out_path + f'/multi_counterfact_identical2_ext_n_{ext_n}.json'
    out_file_path = out_path + f'/multiple_identical_subjects/mcf_multiple_identical_subjects_{ext_n}' + '_{}.json'
    datas1 = load_datas(in_file_path1)
    datas2 = load_datas(in_file_path2)
    make_datas_multiple(datas1, datas2, out_file_path)

    # 2. Sequential Identical Subjects datasets
    in_file_path1 = out_path + f'/multi_counterfact_identical2_ext_n_{ext_n}.json'
    in_file_path2 = out_path + f'/multi_counterfact_identical3_all_105.json'
    in_file_path3 = out_path + f'/multi_counterfact_identical4_all_20.json'
    out_file_path = out_path + '/sequential_identical_subjects/{}/mcf_sequential_identical{}_subjects_{}.json'
    datas_list = [load_datas(in_file_path1), load_datas(in_file_path2), load_datas(in_file_path3)]
    make_datas_sequential(datas_list, out_file_path)





if __name__ == "__main__":
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data'
    out_path = f'{data_dir}/preprocessing'
    
    in_file_path = f'{data_dir}/multi_counterfact.json'
    datas = load_datas(in_file_path)
    run(out_path, datas)

