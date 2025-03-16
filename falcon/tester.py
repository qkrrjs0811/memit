import random
from copy import deepcopy
from tqdm import tqdm

from .falcon_util import *
from .model_editor import *


SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def load_datas(in_file_path: str):
    in_file = open_file(in_file_path, mode='r')
    datas = json.load(in_file)

    print(f'# data_preprocessor.load_datas() datas size : {len(datas)}, in_file_path : {in_file_path}')
    return datas


def get_model_editor(num_edits=100, hparams_fname_suffix='', hparams_mod=None):
    alg_name = 'MEMIT'
    model_name = 'gpt2-xl'
    hparams_fname = model_name + '{}.json'.format(hparams_fname_suffix)
    ds_name = 'mcf'
    # num_edits = 100

    dataset_size_limit = None
    continue_from_run = None
    skip_generation_tests = False
    generation_test_interval = 1
    conserve_memory = False
    dir_name = alg_name
    use_cache = False
    output_hidden_states = False

    model_editor = ModelEditor(
        alg_name, model_name, hparams_fname, ds_name,
        dataset_size_limit, continue_from_run, skip_generation_tests,
        generation_test_interval, conserve_memory, dir_name, num_edits, use_cache, output_hidden_states,
        hparams_mod
    )

    return model_editor


def run():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    identical_group = 1
    num_edits = 1000
    file_name = f'identical{identical_group}_ext_rn_{num_edits}'
    
    in_file_path = f'{data_dir}/multi_counterfact_{file_name}.json'
    datas_subject = load_datas(in_file_path)

    in_file_path = f'{data_dir}/multi_counterfact_{file_name}_sr_swap_post.json'
    datas_relation = load_datas(in_file_path)

    model_editor_subject = get_model_editor(num_edits)
    model_editor_relation = get_model_editor(num_edits, '_test')

    # 기존 subject 데이터로 편집 수행
    model_editor_subject.edit_ext_datas(datas_subject, True, True, True, False, False, False)

    # relation 편집기에 subject 데이터로 결과 확인
    model_editor_relation.edit_ext_datas(datas_subject, True, False, False, False, False, False)

    # subject 편집기의 편집된 모델을 relation 편집기로 넘겨주고, 다시 subject 데이터로 결과 확인
    model_editor_relation._model = model_editor_subject._model
    model_editor_relation.edit_ext_datas(datas_subject, True, False, False, False, False, False)

    # subject 편집된 웨이트를 가진 relation 편집기에 relation 데이터로 편집
    model_editor_relation.edit_ext_datas(datas_relation, True, True, True, False, False, False)

    # 그 다음에 다시 subject 데이터로 결과만 확인
    model_editor_relation.edit_ext_datas(datas_subject, True, False, False, False, False, False)


def run_241201():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    identical_group = 2
    num_edits = 1000
    file_name = f'identical{identical_group}_ext_n_{num_edits}'
    
    in_file_path = f'{data_dir}/multi_counterfact_{file_name}.json'
    datas_subject = load_datas(in_file_path)

    in_file_path = f'{data_dir}/multi_counterfact_{file_name}_sr_swap_post.json'
    datas_relation = load_datas(in_file_path)

    model_editor_subject = get_model_editor(num_edits)
    model_editor_relation = get_model_editor(num_edits, '_test')

    # 기존 subject 데이터로 편집 수행
    model_editor_subject.edit_ext_datas(datas_subject, True, True, True, False, False, False)

    # subject 편집기의 편집된 모델을 relation 편집기로 넘겨주고, 다시 subject 데이터로 결과 확인
    model_editor_relation._model = model_editor_subject._model

    # subject 편집된 웨이트를 가진 relation 편집기에 relation 데이터로 편집
    model_editor_relation.edit_ext_datas(datas_relation, True, True, True, False, False, False)


def run_241204_multiple():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    identical_group = 2
    num_edits = 1000
    file_name = f'identical{identical_group}_ext_n_{num_edits}'
    
    in_file_path = f'{data_dir}/multi_counterfact_{file_name}.json'
    datas_subject = load_datas(in_file_path)

    in_file_path = f'{data_dir}/multi_counterfact_{file_name}_sr_swap_post.json'
    datas_relation = load_datas(in_file_path)

    # 기존 subject 데이터로 편집 수행
    model_editor_subject = get_model_editor(num_edits)
    model_editor_subject.edit_ext_datas(datas_subject, False, True, False, False, False, False)

    layers_list = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8],
                   [5, 6, 7, 8, 9], [6, 7, 8, 9, 10], [7, 8, 9, 10, 11], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13],
                   [10, 11, 12, 13, 14], [11, 12, 13, 14, 15], [12, 13, 14, 15, 16], [13, 14, 15, 16, 17], [14, 15, 16, 17, 18],
                   [15, 16, 17, 18, 19], [16, 17, 18, 19, 20], [17, 18, 19, 20, 21], [18, 19, 20, 21, 22], [19, 20, 21, 22, 23],
                   [20, 21, 22, 23, 24], [21, 22, 23, 24, 25], [22, 23, 24, 25, 26], [23, 24, 25, 26, 27], [24, 25, 26, 27, 28],
                   [25, 26, 27, 28, 29], [26, 27, 28, 29, 30], [27, 28, 29, 30, 31], [28, 29, 30, 31, 32], [29, 30, 31, 32, 33],
                   [30, 31, 32, 33, 34], [31, 32, 33, 34, 35], [32, 33, 34, 35, 36], [33, 34, 35, 36, 37], [34, 35, 36, 37, 38],
                   [35, 36, 37, 38, 39], [36, 37, 38, 39, 40], [37, 38, 39, 40, 41], [38, 39, 40, 41, 42], [39, 40, 41, 42, 43],
                   [40, 41, 42, 43, 44], [41, 42, 43, 44, 45], [42, 43, 44, 45, 46], [43, 44, 45, 46, 47]]

    for layers in tqdm(layers_list):
        hparams_mod = {'layers': layers}
        model_editor_relation = get_model_editor(num_edits, '_test', hparams_mod)

        # subject 편집기의 편집된 모델을 relation 편집기로 복사
        model_editor_relation._model = deepcopy(model_editor_subject._model)

        # subject 편집된 웨이트를 가진 relation 편집기에 relation 데이터로 편집
        model_editor_relation.edit_ext_datas(datas_relation, False, True, True, False, False, False)


def run_241206_sequential():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing/sequential_identical_subjects/each'


    for identical_num, num_edits in zip([4, 3, 2], [5, 35, 500]):
        in_path = f'{data_dir}/identical{identical_num}'

        model_editor_subject = get_model_editor(num_edits)
        model_editor_relation = get_model_editor(num_edits, '_test', {'layers': [26, 27, 28, 29, 30]})

        model_editor_subject._do_eval_org_model = False
        model_editor_subject._do_eval_new_model = False
        model_editor_relation._do_eval_org_model = False
        model_editor_relation._do_eval_new_model = False

        datas_batchs, datas_extend = [], []
        
        for batch_idx in tqdm(range(1, identical_num+1)):
            print(f'### falcon.tester.run_241206_sequential() identical : {identical_num}, batch_size : {num_edits}, batch_idx : {batch_idx}\n')

            in_file_path = in_path + f'/mcf_sequential_identical{identical_num}_subjects_batch{batch_idx}.json'
            datas_subject = load_datas(in_file_path)

            in_file_path = in_path + f'/mcf_sequential_identical{identical_num}_subjects_batch{batch_idx}_sr_swap_post.json'
            datas_relation = load_datas(in_file_path)

            if batch_idx > 1:
                model_editor_subject._model = deepcopy(model_editor_relation._model)

            model_editor_subject.edit_ext_datas(datas_subject, True, True, True, False, False, False)
            model_editor_relation._model = deepcopy(model_editor_subject._model)
            model_editor_relation.edit_ext_datas(datas_relation, True, True, True, False, False, False)

            datas_batchs.append(datas_subject)
            datas_extend.extend(datas_subject)

            if len(datas_batchs) > 1:
                print(f'\n### datas_extend size : {len(datas_extend)}\n')
                for i, datas_batch in enumerate(datas_batchs):
                    print(f'[{i}] batch size : {len(datas_batch)}')
                    model_editor_relation.edit_ext_datas(datas_batch, True, False, False, False, False, False)

                # 테스트 용
                # print(f'\n### datas_extend size : {len(datas_extend)}\n')
                # model_editor_relation._num_edits = len(datas_extend)
                # model_editor_relation.edit_ext_datas(datas_extend, True, False, False, False, False, False)
                # model_editor_relation._num_edits = num_edits
        # break


def run_241219_multiple():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing/multiple_identical_subjects'

    file_name = 'mcf_multiple_identical_subjects_1000_{}:{}{}.json'
    num_edits = 1000
    hparams_mod = {'layers': [26, 27, 28, 29, 30]}
    
    for i in tqdm(range(11)):
        in_file_path = f'{data_dir}/' + file_name.format(i, (10-i), "")
        datas_subject = load_datas(in_file_path)

        in_file_path = f'{data_dir}/' + file_name.format(i, (10-i), "_sr_swap_post")
        datas_relation = load_datas(in_file_path)

        # 기존 subject 데이터로 편집 수행
        model_editor_subject = get_model_editor(num_edits)
        model_editor_subject.edit_ext_datas(datas_subject, False, True, False, False, False, False)

        # subject 편집기의 편집된 모델을 relation 편집기로 복사
        model_editor_relation = get_model_editor(num_edits, '_test', hparams_mod)
        model_editor_relation._model = deepcopy(model_editor_subject._model)

        # subject 편집된 웨이트를 가진 relation 편집기에 relation 데이터로 편집
        model_editor_relation.edit_ext_datas(datas_relation, False, True, True, False, False, False)


def run_250117_multiple_evaluate_matrix():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    # [02_multiple_two_step] : 000 ~ 007
    # file_names = ['multi_counterfact_identical1_ext_rn_1000',
    #               'multi_counterfact_identical2_ext_n_1000',
    #               'multi_counterfact_identical3_all_105',
    #               'multi_counterfact_identical4_all_20']

    # [03_multiple_all_two_step] : 000 ~ 003
    # file_names = ['multi_counterfact_20877',
    #               'multi_counterfact_20877']
    # num_edits_list = [10000, 1000]

    # [04_multiple_identical1,2] : 000 ~ 003
    file_names = ['multi_counterfact_identical1_all_19366',
                  'multi_counterfact_identical2_all_1386']
    num_edits_list = [10000, 1386]

    
    hparams_mod = {'layers': [26, 27, 28, 29, 30]}

    for file_name, num_edits in tqdm(zip(file_names, num_edits_list)):
        # num_edits = int(file_name.split('_')[-1])
        # print(f'file_name : {file_name}, num_edits : {num_edits}')
        # continue

        in_file_path = f'{data_dir}/{file_name}.json'
        datas_subject = load_datas(in_file_path)

        in_file_path = f'{data_dir}/{file_name}_sr_swap_post.json'
        datas_relation = load_datas(in_file_path)

        # 기존 subject 데이터로 편집 수행
        model_editor_subject = get_model_editor(num_edits)
        model_editor_subject.edit_ext_datas(datas_subject, False, True, False, False, False, False)

        # subject 편집기의 편집된 모델을 relation 편집기로 복사
        model_editor_relation = get_model_editor(num_edits, '_test', hparams_mod)
        model_editor_relation._model = deepcopy(model_editor_subject._model)

        # subject 편집된 웨이트를 가진 relation 편집기에 relation 데이터로 편집
        model_editor_relation.edit_ext_datas(datas_relation, False, True, False, False, False, False)


def run_250213_sequential():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing/sequential_identical_subjects/each'


    for identical_num, num_edits in zip([4, 3, 2], [5, 35, 500]):
        in_path = f'{data_dir}/identical{identical_num}'

        model_editor_subject_only = get_model_editor(num_edits)
        model_editor_subject_relation = get_model_editor(num_edits)

        model_editor_subject_only._do_eval_org_model = False
        model_editor_subject_only._do_eval_new_model = False
        model_editor_subject_relation._do_eval_org_model = False
        model_editor_subject_relation._do_eval_new_model = False

        datas_batchs, datas_extend = [], []
        
        for batch_idx in tqdm(range(1, identical_num+1)):
            print(f'### falcon.tester.run_250213_sequential() identical : {identical_num}, batch_size : {num_edits}, batch_idx : {batch_idx}\n')

            in_file_path = in_path + f'/mcf_sequential_identical{identical_num}_subjects_batch{batch_idx}.json'
            datas_subject = load_datas(in_file_path)

            in_file_path = in_path + f'/mcf_sequential_identical{identical_num}_subjects_batch{batch_idx}_sr_swap_post.json'
            datas_relation = load_datas(in_file_path)

            # 기존 방법 적용
            model_editor_subject_only.edit_ext_datas(datas_subject, False, True, False, False, False, False)

            # 제안 방법 적용
            model_editor_subject_relation.set_params_external({'layers': [13, 14, 15, 16, 17]})
            model_editor_subject_relation.edit_ext_datas(datas_subject, False, True, False, False, False, False)
            model_editor_subject_relation.set_params_external({'layers': [26, 27, 28, 29, 30]})
            model_editor_subject_relation.edit_ext_datas(datas_relation, False, True, False, False, False, False)

            # 제안 방법에 대한 배치 단위 성능 측정
            datas_batchs.append(datas_subject)
            datas_extend.extend(datas_subject)

            # if batch_idx > 1:
            print(f'\n### datas_extend size : {len(datas_extend)}\n')
            for i, datas_batch in enumerate(datas_batchs):
                print(f'[{i}] batch size : {len(datas_batch)}')
                if batch_idx == identical_num:
                    model_editor_subject_only._do_eval_org_model = True
                    model_editor_subject_relation._do_eval_org_model = True

                model_editor_subject_only.edit_ext_datas(datas_batch, True, False, False, False, False, False)
                model_editor_subject_relation.edit_ext_datas(datas_batch, True, False, False, False, False, False)
        # break


def run_250214_multiple_only_relation():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing/multiple_identical_subjects'

    file_name = 'mcf_multiple_identical_subjects_1000_{}:{}{}.json'
    num_edits = 1000
    hparams_mod = {'layers': [26, 27, 28, 29, 30]}
    
    for i in tqdm(range(10, -1, -1)):
        in_file_path = f'{data_dir}/' + file_name.format(i, (10-i), "_sr_swap_post")
        # print(f'in_file_path : {in_file_path}')

        # relation으로만 편집 수행
        datas_relation = load_datas(in_file_path)
        model_editor_relation = get_model_editor(num_edits, '_test', hparams_mod)
        model_editor_relation._do_eval_org_model = False
        model_editor_relation._do_eval_new_model = False
        model_editor_relation.edit_ext_datas(datas_relation, False, True, True, False, False, False)


def run_250214_multiple_relation_last_tok():
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    for identical_num, num_edits in tqdm(zip([4, 3, 2], [20, 105, 1000])):
        if identical_num == 2:
            in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_ext_n_{num_edits}' + '{}.json'
        else:
            in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_all_{num_edits}' + '{}.json'
        
        datas_subject = load_datas(in_file_path.format(''))
        datas_relation = load_datas(in_file_path.format('_sr_swap'))

        model_editor_subject_relation = get_model_editor(num_edits)
        model_editor_subject_relation._do_eval_org_model = False
        model_editor_subject_relation._do_eval_new_model = False

        model_editor_subject_relation.set_params_external({'layers': [13, 14, 15, 16, 17]})
        model_editor_subject_relation.edit_ext_datas(datas_subject, False, True, False, False, False, False)
        model_editor_subject_relation.set_params_external({'layers': [26, 27, 28, 29, 30]})
        model_editor_subject_relation.edit_ext_datas(datas_relation, False, True, True, False, False, False)


if __name__ == "__main__":
    # run()
    # run_241201()
    # run_241204_multiple()
    # run_241206_sequential()
    # run_241219_multiple()
    # run_250117_multiple_evaluate_matrix()
    # run_250213_sequential()
    # run_250214_multiple_only_relation()
    run_250214_multiple_relation_last_tok()

