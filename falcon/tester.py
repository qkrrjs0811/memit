import random
from copy import deepcopy

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


def get_model_editor(num_edits=100, hparams_fname_suffix=''):
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

    model_editor = ModelEditor(
        alg_name, model_name, hparams_fname, ds_name,
        dataset_size_limit, continue_from_run, skip_generation_tests,
        generation_test_interval, conserve_memory, dir_name, num_edits, use_cache
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
    model_editor_subject.edit_ext_datas(datas_subject, True, False, False, False)

    # relation 편집기에 subject 데이터로 결과 확인
    model_editor_relation.edit_ext_datas(datas_subject, True, False, False, False, True)

    # subject 편집기의 편집된 모델을 relation 편집기로 넘겨주고, 다시 subject 데이터로 결과 확인
    model_editor_relation._model = model_editor_subject._model
    model_editor_relation.edit_ext_datas(datas_subject, True, False, False, False, True)

    # subject 편집된 웨이트를 가진 relation 편집기에 relation 데이터로 편집
    model_editor_relation.edit_ext_datas(datas_relation, True, False, False, False)

    # 그 다음에 다시 subject 데이터로 결과만 확인
    model_editor_relation.edit_ext_datas(datas_subject, True, False, False, False, True)





if __name__ == "__main__":
    run()

