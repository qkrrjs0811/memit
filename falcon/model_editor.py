import shutil, re, time
import numpy as np
from typing import Tuple, Union, List
from itertools import chain

from .falcon_util import *
from util.globals import *
from util import nethook

from baselines.ft import FTHyperParams, apply_ft_to_model
from baselines.mend import MENDHyperParams, MendRewriteExecutor
from rome import ROMEHyperParams, apply_rome_to_model
from memit import MEMITHyperParams, apply_memit_to_model

from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact, test_batch_prediction, test_generation
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre


from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer
)

ALG_DICT = {
    'FT': (FTHyperParams, apply_ft_to_model),
    'MEND': (MENDHyperParams, MendRewriteExecutor().apply_to_model),
    'ROME': (ROMEHyperParams, apply_rome_to_model),
    'MEMIT': (MEMITHyperParams, apply_memit_to_model)
}


DS_DICT = {
    'mcf': (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    'cf': (CounterFactDataset, compute_rewrite_quality_counterfact),
    'zsre': (MENDQADataset, compute_rewrite_quality_zsre),
}


class FLAGS :
    DO_EVAL_NEW_MODEL = True


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from transformers import  AutoTokenizer
# from .gpt2_model import GPT2LMHeadModel as AutoModelForCausalLM
# from transformers.models.gpt2 import GPT2LMHeadModel as AutoModelForCausalLM

SEED = 7
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# import random
# random.seed(SEED)


import warnings
warnings.filterwarnings('ignore')


class ModelEditor:
    def __init__(self,
                 alg_name: str,
                 model_name: Union[str, Tuple],
                 hparams_fname: str,
                 ds_name: str,
                 dataset_size_limit: int,
                 continue_from_run: str,
                 skip_generation_tests: bool,
                 generation_test_interval: int,
                 conserve_memory: bool,
                 dir_name: str,
                 num_edits=1,
                 use_cache=False,
                 output_hidden_states=False,
                 hparams_mod=None
                ):
        self._alg_name = alg_name
        self._model_name = model_name
        self._hparams_fname = hparams_fname.replace('/', '_')
        self._ds_name = ds_name
        self._dataset_size_limit = dataset_size_limit
        self._continue_from_run = continue_from_run
        self._skip_generation_tests = skip_generation_tests
        self._generation_test_interval = generation_test_interval
        self._conserve_memory = conserve_memory
        self._dir_name = dir_name
        self._num_edits = num_edits
        self._use_cache = use_cache
        self._output_hidden_states = output_hidden_states

        self._params_class, self._apply_algo = ALG_DICT[alg_name]
        self._print_init()

        self._run_dir = ''
        self._case_result_template = ''
        self._check_continue_from_run()

        self._hparams = None
        self._set_params(hparams_mod)

        self._model: AutoModelForCausalLM
        self._tok: AutoTokenizer
        self._weights_copy = None
        self._init_model()

        self._cache_template = None
        self._check_cache()

        self._cnt = 0
        self._performances = [[], [], []]


    def _print_init(self):
        print(f'#################### ModelEditor._print_init() ####################')
        print(f'\talg_name : {self._alg_name}')
        print(f'\tparams_class : {self._params_class}')
        print(f'\tapply_algo : {self._apply_algo}')
        print(f'\tmodel_name : {self._model_name}')
        print(f'\thparams_fname : {self._hparams_fname}')
        print(f'\tds_name : {self._ds_name}')
        print(f'\tdataset_size_limit : {self._dataset_size_limit}')
        print(f'\tcontinue_from_run : {self._continue_from_run}')
        print(f'\tskip_generation_tests : {self._skip_generation_tests}')
        print(f'\tgeneration_test_interval : {self._generation_test_interval}')
        print(f'\tconserve_memory : {self._conserve_memory}')
        print(f'\tdir_name : {self._dir_name}')
        print(f'\tnum_edits : {self._num_edits}')
        print(f'\tuse_cache : {self._use_cache}')
        print(f'\toutput_hidden_states : {self._output_hidden_states}\n\n')
    

    def _check_continue_from_run(self):
        if self._continue_from_run is not None:
            self._run_dir = RESULTS_DIR/self._dir_name/self._continue_from_run
            if not self._run_dir.exists():
                self._continue_from_run = None
        
        if self._continue_from_run is None:
            self._alg_dir = RESULTS_DIR / self._dir_name
            if self._alg_dir.exists():
                id_list = [
                    int(str(x).split('_')[-1])
                    for x in self._alg_dir.iterdir()
                    if str(x).split('_')[-1].isnumeric()
                ]
                run_id = 0 if not id_list else max(id_list) + 1
            else:
                run_id = 0
            self._run_dir = RESULTS_DIR/self._dir_name/f'run_{str(run_id).zfill(3)}'
            self._run_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 결과를 저장하는 파일명 템플릿
        self._case_result_template = str(self._run_dir / '{}_edits-case_{}.json')
        
        print(f'# ModelEditor._check_continue_from_run() Results will be stored at [{self._run_dir}]')
    

    def _set_params(self, hparams_mod: dict=None):
        if self._continue_from_run is None:
            params_path = HPARAMS_DIR / self._alg_name / self._hparams_fname
        else:
            params_path = self._run_dir / 'params.json'
        print(f'# ModelEditor._set_params() params_path : {params_path}')

        self._hparams = self._params_class.from_json(params_path)
        if not (self._run_dir / 'params.json').exists():
            shutil.copyfile(params_path, self._run_dir / 'params.json')
        print(f'# ModelEditor._set_params() [Executing {self._alg_name} with parameters]\n\n{self._hparams}\n')

        # 외부에서 파라미터를 변경해야되는 경우
        if hparams_mod is not None:
            self._hparams.update_from_dict(hparams_mod)
            print(f'# ModelEditor._set_params() [Updated parameters]\n\n{self._hparams}\n')


    def _init_model(self):
        if type(self._model_name) is str:
            print('# ModelEditor._init_model() Instantiating model')
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name,
                                                               output_hidden_states=self._output_hidden_states).cuda()
            self._tok = AutoTokenizer.from_pretrained(self._model_name)
            self._tok.pad_token = self._tok.eos_token
        else:
            self._model, self._tok = self._model_name
            self._model_name = self._model.config._name_or_path
        print(f'\tmodel : {type(self._model)}')
        print(f'\ttokenizer : {type(self._tok)}\n')
    

    def _check_cache(self):
        if self._use_cache:
            self._cache_template = (
                KV_DIR
                / f'{self._model_name.replace("/", "_")}_{self._alg_name}'
                / f'{self._ds_name}_layer_{{}}_clamp_{{}}_case_{{}}.npz')
            print(f'# ModelEditor._check_cache() Will load cache from {self._cache_template}')


    def _check_already(self, record_chunks):
        already_finished = True
        for record in record_chunks:
            if not Path(self._case_result_template.format(self._num_edits, record['case_id'])).exists():
                already_finished = False
                break
        
        return already_finished
    

    def _get_args_memory(self):
        args_conserve_memory = (
            dict(return_orig_weights_device=('cpu' if self._conserve_memory else 'cuda'))
            if self._conserve_memory
            else dict()
        )
        
        etc_args = (
            dict(cache_template=self._cache_template)
            if any(alg in self._alg_name for alg in ['ROME', 'MEMIT'])
            else dict()
        )

        return args_conserve_memory, etc_args


    def _generate(self, prompts: List[str], top_k, max_out_len):
        outputs = self._model.generate(**self._tok(prompts, return_tensors='pt').to('cuda'),
                                  top_k=top_k, max_length=max_out_len,
                                  do_sample=False, num_beams=1,
                                  pad_token_id=self._tok.eos_token_id)
        
        gen_texts = str(self._tok.decode(outputs[0], skip_special_token=True))
        return re.sub(r'[\t\n ]+', ' ', gen_texts).strip()


    def _predict_all(self, model, tok, records, top_k=1, max_out_len=100, do_print=False, prefix='', gen_prompt_size=1, out_dir=''):
        performance = 0

        for idx, record in enumerate(records):
            cnt = idx + 1 if 'extend' in prefix else self._cnt + idx + 1
            ret = self._predict(model, tok, record, top_k, max_out_len, do_print, f'[{cnt}] {prefix}', gen_prompt_size)
            
            if len(out_dir) > 0:
                logits = ret['logits']
                self._write_logits(out_dir.format(prefix), logits, self._tok, idx)
            
            if is_true(ret['targets_correct']):
                performance += 1
        
        performance_p = performance / len(records)
        print(f'# ModelEditor._predict_all() # [{prefix}] performance : {performance}/{len(records)} = {performance_p}\n')

        if 'edited' == prefix:
            self._performances[0].append(performance_p)
        elif 'extend' == prefix:
            self._performances[1].append(performance_p)
        elif 'restored' in prefix:
            self._performances[2].append(performance_p)


    '''
        - 단일 데이터에 대해서 결과를 확인하기 위해 만든 임시 함수
            - eval_utils_counterfact.compute_rewrite_quality_counterfact() 참고
    '''
    def _predict(self, model, tok, record, top_k=1, max_out_len=100, do_print=False, prefix='', gen_prompt_size=1):
        case_id = record['case_id']
        subject, target_new, target_true = (
            record['requested_rewrite'][x] for x in ['subject', 'target_new', 'target_true']
        )
        rewrite_prompts = [record['requested_rewrite']['prompt'].format(subject)]
        generation_prompts = record['generation_prompts']
        if len(generation_prompts) > gen_prompt_size:
            generation_prompts = generation_prompts[:gen_prompt_size]

        prob_prompts = [rewrite_prompts]
        which_correct = [[0 for _ in range(len(rewrite_prompts))]]

        # probs의 값은 log_softmax의 음수를 취한 값으로 작은 값이 실제 확률 분포에서 큰 값이다.
        probs, targets_correct, logits = test_batch_prediction(model, tok,
            list(chain(*prob_prompts)),
            list(chain(*which_correct)),
            target_new['str'],
            target_true['str'],
            prefix
        )

        gen_texts_r_prompt = self._generate(rewrite_prompts, top_k, max_out_len)
        gen_texts_g_prompt = self._generate(generation_prompts, top_k, max_out_len)

        if do_print:
            print(f'# ModelEditor._predict() [ {prefix} ] case_id : {case_id}')
            print(f'# ModelEditor._predict() [ {prefix} ] subject : {subject}\n')

            print(f'# ModelEditor._predict() [ {prefix} ] probs : {probs}')
            print(f'# ModelEditor._predict() [ {prefix} ] targets_correct : {targets_correct}\n')
                
            print(f'# ModelEditor._predict() [ {prefix} ] rewrite_prompts : {rewrite_prompts}')
            print(f'# ModelEditor._predict() [ {prefix} ] gen_texts_r_prompt : {gen_texts_r_prompt}\n')
                
            print(f'# ModelEditor._predict() [ {prefix} ] generation_prompts : {generation_prompts}')
            print(f'# ModelEditor._predict() [ {prefix} ] gen_texts_g_prompt : {gen_texts_g_prompt}\n\n')
        
        return {'subject': subject, 'probs': probs, 'targets_correct': targets_correct,
                'gen_texts_r_prompt': gen_texts_r_prompt, 'gen_texts_g_prompt': gen_texts_g_prompt,
                'logits': logits}


    def _write_logits(self, out_dir, logits, tok, idx):
        decoded_toks = {}
        tok_len = len(logits[0][0])
        
        for i in range(tok_len):
            decoded_toks[i] = tok.decode(i)

        for i in range(logits.size(0)):
            j_len = logits[i].size(0)
            
            for j in range(j_len):
                if j < j_len-2:
                    continue

                out_file_path = f'{out_dir}/logits_decoded_{idx}_[{i}][{j}].txt'
                out_file = open_file(out_file_path, mode='w')

                probs = torch.nn.functional.softmax(logits[i, j, :], dim=0)
                write_dict = {}
                
                for k in range(len(probs)):
                    write_dict[f'{k}\t{decoded_toks[k]}'] = probs[k].item()
                                
                write_dict = dict(sorted(write_dict.items(), key=lambda item:item[1], reverse=True))
                items = write_dict.items()
                
                for item in items:
                    out_file.write(f'{item[0]}\t{item[1]}\n')
                out_file.close()
    

    def _load_data(self):
        self._snips = AttributeSnippets(DATA_DIR) if not self._skip_generation_tests else None
        self._vec = get_tfidf_vectorizer(DATA_DIR) if not self._skip_generation_tests else None

        self._ds_class, self._ds_eval_method = DS_DICT[self._ds_name]
        print(f'# ModelEditor._load_data() ds_class : {self._ds_class}')
        print(f'# ModelEditor._load_data() ds_eval_method : {self._ds_eval_method}\n')


    def load_data(self):
        self._load_data()
        self._ds = self._ds_class(DATA_DIR, tok=self._tok, size=self._dataset_size_limit)
    

    def _evaluate(self, model, tok, records, exec_time):
        case_ids = [record['case_id'] for record in records]
        gen_test_vars = [self._snips, self._vec]

        start = time.time()
        for record in records:
            out_file = Path(self._case_result_template.format(self._num_edits, record['case_id']))
            if out_file.exists():
                print(f'# ModelEditor._evaluate() skipping [{out_file}] already exists')
                continue

            metrics = {
                "case_id": record["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": self._num_edits,
                "requested_rewrite": record["requested_rewrite"],
                "time": exec_time,
                "post": self._ds_eval_method(
                    model,
                    tok,
                    record,
                    *(
                        gen_test_vars
                        if record["case_id"] % self._generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        eval_time = time.time() - start
        print(f'# ModelEditor._evaluate() Evaluation took : {eval_time}\n\n')


    def restore_weights(self):
        with torch.no_grad():
            for k, v in self._weights_copy.items():
                nethook.get_parameter(self._model, k)[...] = v.to('cuda')
            
            self._weights_copy = None
            print(f'# ModelEditor.restore_weights() weights restored\n')


    def edit_ext_datas(self, datas, do_org_test=True, do_edit=True, do_edit_test=True, do_extend_test=True, do_restore=False, do_restore_test=False, do_print=True):
        self._load_data()
        self._ds = datas
        self.edit(do_org_test, do_edit, do_edit_test, do_extend_test, do_restore, do_restore_test, do_print)


    def edit(self, do_org_test=True, do_edit=True, do_edit_test=True, do_extend_test=True, do_restore=False, do_restore_test=False, do_print=True):
        if self._num_edits > 1:
            assert self._ds_name != 'cf', f'{self._ds_name} does not support multiple edits'
        
        
        ##### flag 출력 #####
        print(f'\n# ModelEditor.edit() args')
        print(f'\tdo_org_test : {do_org_test}, do_edit : {do_edit}, do_edit_test : {do_edit_test}, do_extend_test : {do_extend_test}')
        print(f'\tdo_restore : {do_restore}, do_restore_test : {do_restore_test}, do_print : {do_print}\n')

        # 누적 실험을 위한 리스트
        record_chunks_ext, case_ids_ext = [], []

        for record_chunks in chunks(self._ds, self._num_edits):
            # 기존에 작업한 데이터인지 확인
            if self._check_already(record_chunks):
                continue

            # cpu, gpu, cache 설정
            args_conserve_memory, etc_args = self._get_args_memory()

            # 확인을 위해, 한 번에 편집되는 전체 case_id list 생성
            case_ids = [record['case_id'] for record in record_chunks]
            print(f'######################################## case_ids : {case_ids} ########################################')

            
            # (1) 기존 결과 확인
            if do_org_test:
                self._predict_all(self._model, self._tok, record_chunks, do_print=do_print, prefix='org')

            # (2) 모델 편집
            '''
                - ROME : rome_main.apply_rome_to_model()
                - MEMIT : memit_main.apply_memit_to_model()
            '''
            if do_edit:
                start = time.time()
                edited_model, self._weights_copy = self._apply_algo(
                    self._model, self._tok,
                    [{'case_id': record['case_id'], **record['requested_rewrite']} for record in record_chunks],
                    self._hparams, copy=False, return_orig_weights=True,
                    **args_conserve_memory, **etc_args
                )
                exec_time = time.time() - start
                print(f'# ModelEditor.edit() Execution took : {exec_time}\n\n')

                # 변경된 모델에 대해서 성능 평가
                if FLAGS.DO_EVAL_NEW_MODEL:
                    self._evaluate(edited_model, self._tok, record_chunks, exec_time)

            # (3) 변경된 결과 확인
            if do_edit_test:
                self._predict_all(edited_model, self._tok, record_chunks, do_print=do_print, prefix='edited')


            # (4) 현재까지의 전체 데이터 테스트
            if do_extend_test:
                record_chunks_ext.extend(record_chunks)
                case_ids_ext.extend(case_ids)

                out_dir = '/home/nlpshlee/dev_env/git/repos/memit/logs/logs_{}'
                if len(case_ids_ext) <= 10:
                    out_dir += f'/id_{case_ids_ext}'
                else:
                    out_dir += f'/id_{case_ids_ext[:10]}...{case_ids_ext[-1]}'

                print('\n\n######################################## extend ########################################\n')
                # self._predict_all(edited_model, self._tok, record_chunks_ext, do_print=do_print, prefix='extend', out_dir=out_dir)
                self._predict_all(edited_model, self._tok, record_chunks_ext, do_print=do_print, prefix='extend')
            
            # (5) 편집 이전의 weight로 복원 및 테스트
            if do_restore:
                # 편집된 weight를 다시 원래 weight로 복원
                self.restore_weights()

                if do_restore_test:
                    if not do_extend_test:
                        print('\n\n######################################## restored ########################################\n')
                        self._predict_all(self._model, self._tok, record_chunks, do_print=do_print, prefix='restored')
                    # else:
                    #     print('\n\n######################################## restored_extend ########################################\n')
                    #     self._predict_all(self._model, self._tok, record_chunks_ext, do_print=do_print, prefix='restored_extend', out_dir=out_dir)

            self._cnt += len(record_chunks)
            print(f'[{self._cnt}] edit finish\n\n')

            self._print_performance()

            # if self._cnt >= 100:
            #     break


    def _print_performance(self):
        print(f'\n# ModelEditor.print_performance()')
        print(f'\tedited : {self._performances[0]}')
        print(f'\tedited (avg) : {np.mean(self._performances[0])}')
        print(f'\textend : {self._performances[1]}')
        print(f'\textend (avg) : {np.mean(self._performances[1])}')
        print(f'\trestored : {self._performances[2]}')
        print(f'\trestored (avg) : {np.mean(self._performances[2])}\n')

