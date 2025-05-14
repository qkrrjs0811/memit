import argparse
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


def get_model_editor(num_edits=100, hparams_fname_suffix='', hparams_mod=None, identical_num=1, merged_model_path=None, pruning=False):
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
        hparams_mod, identical_num, merged_model_path, pruning
    )

    return model_editor


def edit_and_save():
    # 현재 목적: 각 batch를 독립적으로 편집하고 모델만 저장 (평가는 이후 별도 실행)

    home_dir = '/home/albert0811/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    for identical_num, num_edits in zip([2], [250]):
        # identical_num: 총 batch의 개수
        # num_edits: 하나의 batch에 들어간 데이터 개수

        model_editor = get_model_editor(num_edits)

        model_editor._do_eval_org_model = False
        model_editor._do_eval_new_model = False
        
        # === Normal Edit ===
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)

        # # === Sequential Edit ===
        # in_file_path = f'{data_dir}/mcf_sequential_identical{identical_num}_subjects_all.json'
        # datas_subject = load_datas(in_file_path)


        # # === Multiple Edit ===
        # size = identical_num * num_edits
        # if identical_num == 2:
        #     in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_ext_n_{size}.json'
        # else:
        #     in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_all_{size}.json'
        # datas_subject = load_datas(in_file_path)


        print("\n[SAVE] 각 batch별로 편집한 모델 독립적으로 저장중...")
        model_editor.edit_ext_datas(
            datas_subject,
            do_org_test=False,
            do_edit=True,
            do_edit_test=False,
            do_extend_test=False,
            do_restore=True,
            do_restore_test=False,
            do_print=True,
            do_save=True  # 저장 목적
        )
        
        # Origin Edit method 평가 수행
        model_editor_org = get_model_editor(num_edits)
        model_editor_org._do_eval_org_model = False
        model_editor_org._do_eval_new_model = False

        print("\n[Test] 기존 Editing 방법에 대해 평가 수행 중...")
        model_editor_org.edit_ext_datas(datas_subject, False, True, False, True, False, False)


def merge_test(model_name):

    # 경로 설정
    home_dir = '/home/albert0811/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'
    merged_model_path = Path(f"{home_dir}/merged/{model_name}")

    # 병합된 모델 디렉토리 존재 여부 확인
    if not merged_model_path.exists():
        print(f"[Error] 병합된 모델 디렉토리가 존재하지 않습니다: {merged_model_path}")
        return

    print(f"[Test] 병합된 모델 로드 중... from {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_model_path),
        torch_dtype=torch.float32
    ).cuda()
    tok = AutoTokenizer.from_pretrained(str(merged_model_path))
    tok.pad_token = tok.eos_token


    # === Normal Edit ===
    # 실험용 데이터 로드 및 평가 실행
    for identical_num, num_edits in zip([2], [250]):
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)


        # Merge한 모델 평가 수행
        model_editor_merging = get_model_editor(num_edits)
        model_editor_merging._model = model
        model_editor_merging._tok = tok
        model_editor_merging._do_eval_org_model = False
        model_editor_merging._do_eval_new_model = True

        print("[Test] 병합된 모델에 대해 평가 수행 중...")
        model_editor_merging.edit_ext_datas(
            datas_subject,
            do_org_test=False,
            do_edit=False,
            do_edit_test=False,
            do_extend_test=True,  # 🔹 평가 목적
            do_restore=False,
            do_restore_test=False,
            do_print=False,
            do_save=False
        )
    

    # # === Sequential Edit ===
    # # 실험용 데이터 로드 및 평가 실행
    # for identical_num, num_edits in zip([4], [5]):
    #     in_file_path = f'{data_dir}/mcf_sequential_identical{identical_num}_subjects_all.json'
    #     datas_subject = load_datas(in_file_path)


    #     # Merge한 모델 평가 수행
    #     model_editor_merging = get_model_editor(num_edits)
    #     model_editor_merging._model = model
    #     model_editor_merging._tok = tok
    #     model_editor_merging._do_eval_org_model = False
    #     model_editor_merging._do_eval_new_model = True

    #     print("[Test] 병합된 모델에 대해 평가 수행 중...")
    #     model_editor_merging.edit_ext_datas(
    #         datas_subject,
    #         do_org_test=False,
    #         do_edit=False,
    #         do_edit_test=False,
    #         do_extend_test=True,  # 🔹 평가 목적
    #         do_restore=False,
    #         do_restore_test=False,
    #         do_print=False,
    #         do_save=False
    #     )


    # # === Multiple Edit ===
    # # 실험용 데이터 로드 및 평가 실행
    # for identical_num, num_edits in zip([2], [500]):
    #     size = identical_num * num_edits
    #     if identical_num == 2:
    #         in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_ext_n_{size}.json'
    #     else:
    #         in_file_path = f'{data_dir}/multi_counterfact_identical{identical_num}_all_{size}.json'
    #     datas_subject = load_datas(in_file_path)


    #     # Merged된 모델 평가 수행
    #     model_editor = get_model_editor(num_edits)
    #     model_editor._model = model
    #     model_editor._tok = tok
    #     model_editor._do_eval_org_model = False
    #     model_editor._do_eval_new_model = True

    #     print("[Test] 병합된 모델에 대해 평가 수행 중...")
    #     model_editor.edit_ext_datas(
    #         datas_subject,
    #         do_org_test=False,
    #         do_edit=False,
    #         do_edit_test=False,
    #         do_extend_test=True,  # 🔹 평가 목적
    #         do_restore=False,
    #         do_restore_test=False,
    #         do_print=False,
    #         do_save=False
    #     )


def kcc_edit_and_save(identical_nums, num_edits_list):
    home_dir = '/home/albert0811/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        model_editor = get_model_editor(num_edits, identical_num=identical_num)

        model_editor._do_eval_org_model = False
        model_editor._do_eval_new_model = False
        
        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)

        print("\n[SAVE] 각 batch별로 모델 편집 및 독립적으로 저장중...")
        model_editor.edit_ext_datas(
            datas_subject,
            do_org_test=False,
            do_edit=True,
            do_edit_test=True,
            do_extend_test=False,
            do_restore=True,
            do_restore_test=False,
            do_print=True,
            do_save=True,  # 저장 목적
            log_independent=True,
            log_merged=False,
            log_memit=False
        )


def kcc_merge_test(identical_nums, num_edits_list):
    # 경로 설정
    home_dir = '/home/albert0811/dev_env/git/repos/memit'
    data_dir = f'{home_dir}/data/preprocessing'
    merged_dir = Path(f"{home_dir}/merged")

    for identical_num, num_edits in zip(identical_nums, num_edits_list):
        # Origin Edit method 평가 수행
        model_editor_org = get_model_editor(num_edits, identical_num=identical_num)
        model_editor_org._do_eval_org_model = False
        model_editor_org._do_eval_new_model = False

        size = identical_num * num_edits
        in_file_path = f'{data_dir}/normal/mcf_sampled_{size}.json'
        datas_subject = load_datas(in_file_path)

        # /logs 디렉토리에 memit_{identical_num}_{num_edits}.txt와 동일한 파일이 존재하는지 확인
        log_file_path = f"./logs/memit_{identical_num}_{num_edits}.txt"
        print("#########################")
        print(f"log_file_path: {log_file_path}")
        if os.path.exists(log_file_path):
            print(f"[Warning] 동일한 이름의 로그 파일이 이미 존재합니다: {log_file_path}")
            print("기존 Editing 방법에 대한 평가를 건너뛰고 병합된 모델 디렉토리 탐색으로 넘어갑니다.")
        else:
            print("\n[Test] 기존 Editing 방법에 대해 평가 수행 중...")
            model_editor_org.edit_ext_datas(
                datas_subject,
                do_org_test=False,
                do_edit=True,
                do_edit_test=True,
                do_extend_test=False,
                do_restore=False,
                do_restore_test=False,
                do_print=True,
                do_save=False,
                log_independent=False,
                log_merged=False,
                log_memit=True
            )

        # 병합된 모델 디렉토리 탐색
        for merged_model_path in merged_dir.iterdir():
            if not merged_model_path.is_dir():
                continue

            # /logs 디렉토리에 동일한 이름의 txt 파일이 존재하는지 확인
            log_file_path = f"./logs/{merged_model_path.name}.txt"
            print("#########################")
            print(f"log_file_path: {log_file_path}")
            if os.path.exists(log_file_path):
                print(f"[Warn] 동일한 이름의 로그 파일이 이미 존재합니다: {log_file_path}")
                print("다음 경로로 넘어갑니다.")
                continue

            print(f"[Test] 병합된 모델 로드 중... from {merged_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                str(merged_model_path),
                torch_dtype=torch.float32
            ).cuda()
            tok = AutoTokenizer.from_pretrained(str(merged_model_path))
            tok.pad_token = tok.eos_token


            # 모델 편집기 초기화
            if "gpt2_xl" in str(merged_model_path):
                model_editor_merging = get_model_editor(int(num_edits/2), identical_num=int(identical_num*2), merged_model_path=merged_model_path, pruning=True)
            else:
                model_editor_merging = get_model_editor(num_edits, identical_num=identical_num, merged_model_path=merged_model_path)


            # Merge한 모델 평가 수행
            model_editor_merging._model = model
            model_editor_merging._tok = tok
            model_editor_merging._do_eval_org_model = False
            model_editor_merging._do_eval_new_model = False

            print("[Test] 병합된 모델에 대해 평가 수행 중...")
            model_editor_merging.edit_ext_datas(
                datas_subject,
                do_org_test=False,
                do_edit=False,
                do_edit_test=True,
                do_extend_test=False,  # 🔹 평가 목적
                do_restore=False,
                do_restore_test=False,
                do_print=True,
                do_save=False,
                log_independent=False,
                log_merged=True,
                log_memit=False
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--identical_nums', type=int, nargs='+', help='List of identical numbers for batches')
    parser.add_argument('--num_edits_list', type=int, nargs='+', help='List of number of edits per batch')

    args = parser.parse_args()

    
    # 독립적으로 편집 후 저장
    # kcc_edit_and_save(args.identical_nums, args.num_edits_list)

    # memit 환경에서 실행
    kcc_merge_test(args.identical_nums, args.num_edits_list)
