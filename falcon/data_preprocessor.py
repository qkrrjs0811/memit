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


def sr_swap(prompt: str, subject: str):
    eojs = prompt.split()

    idx = -1
    eoj, suffix = '#%#', '#%#'

    for i, eoj in enumerate(eojs):
        idx = i
        if '{}' == eoj:
            suffix = ''
        if '{}' in eoj:
            suffix = eoj[2:]
        else:
            idx = -1
        
        if idx != -1:
            break
    
    if idx == -1:
        print(f'### sr_swap() error : prompt = {prompt}, subject = {subject}')
        return None, None
    elif idx == 0:
        relation = ' '.join(eojs[1:])
        prompt_sr_swap = f'{subject}{suffix}' + ' {}'
    else:
        left = ' '.join(eojs[:idx])
        right = ' '.join(eojs[idx+1:])

        if ('play?' in right) or (left == 'The headquarter of' and right == 'is located in'):
            relation = right
            prompt_sr_swap = f'{left} {subject}{suffix}' + ' {}'
        elif len(left) < len(right):
            relation = right
            prompt_sr_swap = f'{left} {subject}{suffix}' + ' {}'
        elif len(left) > len(right):
            relation = left
            prompt_sr_swap = '{} ' + f'{subject}{suffix} {right}'
        else:
            return None, None
    
    return prompt_sr_swap, relation


def rm_relation_last(prompt: str, relation: str):
    prompt_eojs = prompt.split()
    relation_eojs = relation.split()

    if 1 < len(relation_eojs):
        for i, prompt_eoj in enumerate(prompt_eojs):
            if prompt_eoj == '{}':
                prompt = ' '.join(prompt_eojs[:i+1])
                relation = ' '.join(relation_eojs[:-1])

                prompt += f' {relation_eojs[-1]}'

                if i+1 < len(prompt_eojs):
                    prompt_other = ' '.join(prompt_eojs[i+1:])
                    prompt += f' {prompt_other}'

                break

    return prompt, relation


def make_datas_sr_swap(datas, out_path: str):
    datas_sr_swap, datas_sr_swap_post = [], []

    for data in datas:
        prompt = data['requested_rewrite']['prompt']
        subject = data['requested_rewrite']['subject']

        prompt_sr_swap, relation = sr_swap(prompt, subject)
        prompt_sr_swap_post, relation_post = rm_relation_last(prompt_sr_swap, relation)
        
        ''' 확인용 코드'''
        text1 = prompt.format(subject)
        text2 = prompt_sr_swap.format(relation)
        text3 = prompt_sr_swap_post.format(relation_post)
        if not (text1 == text2 and text2 == text3):
            print(f'prompt : {prompt}')
            print(f'subject : {subject}\n')
            print(f'prompt_sr_swap : {prompt_sr_swap}')
            print(f'relation : {relation}\n')
            print(f'prompt_sr_swap_post : {prompt_sr_swap_post}')
            print(f'relation_post : {relation_post}\n')
            sys.exit(-1)
        
        ''' 변경된 데이터 생성 '''
        data_sr_swap = deepcopy(data)
        data_sr_swap['requested_rewrite']['prompt'] = prompt_sr_swap
        data_sr_swap['requested_rewrite']['subject'] = relation
        data_sr_swap_post = deepcopy(data)
        data_sr_swap_post['requested_rewrite']['prompt'] = prompt_sr_swap_post
        data_sr_swap_post['requested_rewrite']['subject'] = relation_post

        datas_sr_swap.append(data_sr_swap)
        datas_sr_swap_post.append(data_sr_swap_post)
    
    if out_path is not None:
        write_datas(out_path.format('_sr_swap'), datas_sr_swap)
        write_datas(out_path.format('_sr_swap_post'), datas_sr_swap_post)

    return datas_sr_swap, datas_sr_swap_post


def make_datas_sr_both(datas, out_path: str):
    datas_sr_both = []

    for data in datas:
        prompt = data['requested_rewrite']['prompt']
        subject = data['requested_rewrite']['subject']

        prompt_sr_swap, relation = sr_swap(prompt, subject)
        prompt_sr_swap_post, relation_post = rm_relation_last(prompt_sr_swap, relation)

        ''' 변경된 데이터 생성 '''
        data_sr_both = deepcopy(data)
        data_sr_both['requested_rewrite']['rel_prompt'] = prompt_sr_swap_post
        data_sr_both['requested_rewrite']['relation'] = relation_post

        datas_sr_both.append(data_sr_both)
    
    if out_path is not None:
        write_datas(out_path.format('_sr_both'), datas_sr_both)


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





def run1(datas, out_path):
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


def run2(file_names: list, out_path: str):
    for file_name in file_names:
        datas = load_datas(os.path.join(out_path, file_name.format('')))
        make_datas_sr_swap(datas, os.path.join(out_path, file_name))


def run3(in_path: str):
    in_file_paths = get_file_paths(in_path, True)

    for in_file_path in in_file_paths:
        if not in_file_path.endswith('_all.json'):
            file_path = in_file_path.split('.')[0].strip() + '{}.json'
            datas = load_datas(file_path.format(''))
            make_datas_sr_swap(datas, file_path)


def run4(file_names: list, out_path: str):
    for file_name in file_names:
        datas = load_datas(os.path.join(out_path, file_name.format('')))
        make_datas_sr_both(datas, os.path.join(out_path, file_name))





if __name__ == "__main__":
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data'
    out_path = f'{data_dir}/preprocessing'
    
    in_file_path = f'{data_dir}/multi_counterfact.json'
    # datas = load_datas(in_file_path)
    # run1(datas, out_path)

    file_names = ['multi_counterfact_identical1_ext_rn_1000{}.json',
                  'multi_counterfact_identical2_ext_n_1000{}.json',
                  'multi_counterfact_identical3_all_105{}.json',
                  'multi_counterfact_identical4_all_20{}.json']
    # run2(file_names, out_path)
    # run4(file_names, out_path)

    file_names = ['multi_counterfact_1000{}.json',
                  'multi_counterfact_10000{}.json']
    run2(file_names, out_path)

    in_path = f'{data_dir}/preprocessing/sequential_identical_subjects/each'
    # run3(in_path)

    file_names = ['mcf_multiple_identical_subjects_1000_10:0{}.json',
                  'mcf_multiple_identical_subjects_1000_9:1{}.json',
                  'mcf_multiple_identical_subjects_1000_8:2{}.json',
                  'mcf_multiple_identical_subjects_1000_7:3{}.json',
                  'mcf_multiple_identical_subjects_1000_6:4{}.json',
                  'mcf_multiple_identical_subjects_1000_5:5{}.json',
                  'mcf_multiple_identical_subjects_1000_4:6{}.json',
                  'mcf_multiple_identical_subjects_1000_3:7{}.json',
                  'mcf_multiple_identical_subjects_1000_2:8{}.json',
                  'mcf_multiple_identical_subjects_1000_1:9{}.json',
                  'mcf_multiple_identical_subjects_1000_0:10{}.json']
    out_path = f'{data_dir}/preprocessing/multiple_identical_subjects'
    # run2(file_names, out_path)

