import time
from SPARQLWrapper import SPARQLWrapper, JSON


from .falcon_util import *
from .model_editor import *


seed = 7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def add_dict(in_dict: dict, key: str, value: str):
    if not key in in_dict.keys():
        in_dict[key] = set()
    
    value_set = in_dict[key]
    value_set.add(value)


def sorted_value(in_dict: dict):
    sorted_dict = {}
    for key in in_dict.keys():
        value_list = list(in_dict[key])
        value_list.sort()

        sorted_dict[key] = value_list
    
    return sorted_dict


def write_dict(out_file_path, out_dict: dict, delim_key='\t'):
    sorted_dict = sorted_dict_key(out_dict)

    out_file = open_file(out_file_path, mode='w')
    v_cnt_all = 0

    for key, value_set in sorted_dict.items():
        value_list = sorted(list(value_set))
        value_size = len(value_list)
        value_str = delim_key.join(value_list)

        out_file.write(f'{key}{delim_key}{value_size}{delim_key}{value_str}\n')

        v_cnt_all += value_size

    out_file.close()
    print(f'find_with_wiki.write_dict() out_path : {out_file_path}, size : {len(out_dict)}, v_cnt_all : {v_cnt_all}')


def generate(model, tok, prompts: List[str], top_k, max_out_len):
    outputs = model.generate(**tok(prompts, return_tensors='pt').to('cuda'), 
                             top_k=top_k, max_length=max_out_len, 
                             do_sample=False, num_beams=1, 
                             pad_token_id=tok.eos_token_id)

    text_gen = str(tok.decode(outputs[0], skip_special_token=True))
    text_gen = re.sub(r'[\t\n ]+', ' ', text_gen).strip()

    if text_gen.startswith('<|begin_of_text|>'):
        text_gen = text_gen[len('<|begin_of_text|>'):]
            
    return text_gen, outputs


def rm_prompt(prompt, text_gen: str):
    prompt_len = len(prompt)
    if prompt == text_gen[:prompt_len]:
        return text_gen[prompt_len:].strip()

    return ''


def check_start_target(target: str, text: str):
    target = target.lower()
    text = text.lower()

    if target == text:
        return True
    elif text.startswith(f'{target} '):
        return True
    
    return False


def find_triple_with_wiki(relation_id, object_id, limit=1):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.addCustomHttpHeader("User-Agent", "Mozilla/5.0")

    query = f"""
        SELECT DISTINCT ?subject ?subjectLabel ?property ?propertyLabel WHERE {{
            VALUES (?predicate) {{(wdt:{relation_id})}} 
            VALUES (?item) {{(wd:{object_id})}} 
            ?subject ?predicate ?item .
            ?property wikibase:directClaim ?predicate .
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """
    # print(f'query : {query}')

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query
    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f'\n\nfind_with_wiki.find_triple_with_wiki() error msg : {e}\n\n')
        return None, None

    find_subject_list, find_relation_list = [], []

    for result in results["results"]["bindings"]:
        subject = result.get("subjectLabel", {}).get("value", result.get("subject", {}).get("value", "No Label"))
        relation = result.get("propertyLabel", {}).get("value", result.get("property", {}).get("value", "No Label"))

        def _post(ent: str):
            return ent.replace('...', '').strip()
        
        find_subject_list.append(_post(subject))
        find_relation_list.append(_post(relation))
    
    return find_subject_list, find_relation_list


def convert_to_json(in_file_path, out_file_path, select_ids: set):
    in_file = open_file(in_file_path, mode='r')
    
    ents = []
    cnt = 0

    while 1:
        line = in_file.readline()
        if not line:
            break

        temp = line.split('\t')
        target_id = temp[0]
        relation_id = temp[1]
        prompt = temp[2]
        subject = temp[3]
        target = temp[5]

        if len(select_ids) > 0 and (not target_id in select_ids):
            continue

        ent = {'case_id': cnt, 'pararel_idx': -1, 'requested_rewrite': {'prompt': prompt,
                                                                        'relation_id': relation_id,
                                                                        'target_new': {'str': '', 'id': ''},
                                                                        'target_true': {'str': target, 'id': target_id},
                                                                        'subject': subject},
               'paraphrase_prompts': [], 'neighborhood_prompts': [], 'attribute_prompts': [], 'generation_prompts': []}

        ents.append(ent)
        cnt += 1
    in_file.close()

    write_json_to_file(ents, out_file_path)
    print(f'find_with_wiki.convert_to_json() {in_file_path} -> {out_file_path}\n')


RELATION_DICT_MQUAKE = {
		  "P17": ["{} is located in the country of"],
		  "P19": ["{} was born in the city of"],
		  "P20": ["{} died in the city of"],
		  "P26": ["{} is married to"],
		  "P30": ["{} is located in the continent of"],
		  "P36": ["The capital of {} is"],
		  "P40": ["{}'s child is"],
		  "P50": ["The author of {} is"],
		  "P69": ["The univeristy where {} was educated is"],
		  "P106": ["{} works in the field of"],
		  "P112": ["{} was founded by"],
		  "P127": ["{} is owned by"],
		  "P131": ["{} is located in"],
		  "P136": ["The type of music that {} plays is"],
		  "P159": ["The headquarters of {} is located in the city of"],
		  "P170": ["{} was created by"],
		  "P175": ["{} was performed by"],
		  "P176": ["The company that produced {} is"],
		  "P264": ["{} is represented by"],
		  "P276": ["{} is located in"],
		  "P407": ["{} was written in the language of"],
		  "P413": ["{} plays the position of"],
		  "P495": ["{} was created in the country of"],
		  "P740": ["{} was founded in the city of"],
		  "P800": ["{} is famous for"],
		  "P364": ["The original language of {} is"],
		  "P1412": ["{} speaks the language of"],
		  "P27": ["{} is a citizen of"],
		  "P937": ["{} worked in the city of"],
		  "P37": ["The official language of {} is"],
		  "P140": ["{} is affiliated with the religion of"],
		  "P178": ["{} was developed by"],
		  "P108": ["{} is employed by"],
		  "P641": ["{} is associated with the sport of"],
		  "P190": ["The twinned administrative body of {} is"],
		  "P463": ["{} is a member of"],
		  "P449": ["The origianl broadcaster of {} is"],
		  "P35": ["The name of the current head of state in {} is"],
		  "P286": ["The head coach of {} is"],
		  "P1308": ["The {} is"],
		  "P488": ["The chairperson of {} is"],
		  "P169": ["The chief executive officer of {} is"],
		  "P1037": ["The director of {} is"],
		  "P6": ["The name of the current head of the {} government is"]
}


class FindWithWiki:
    def __init__(self, model, tok):
        self._model = model
        self._tok = tok

        self.relation_dict = {}
        self.target_true_dict = {}
        self.target_new_dict = {}
        self.target_new_relation_dict = {}


    def _generate(self, prompts: List[str], top_k, max_out_len):
        text_gen, outputs = generate(self._model, self._tok, prompts, top_k, max_out_len)
        return text_gen


    def _sorted_dicts(self):
        self.relation_dict = sorted_value(self.relation_dict)
        self.target_true_dict = sorted_value(self.target_true_dict)
        self.target_new_dict = sorted_value(self.target_new_dict)
        self.target_new_relation_dict = sorted_value(self.target_new_relation_dict)
    

    def _write_dicts(self, out_dir):
        if len(out_dir) != 0:
            out_file_path = f'{out_dir}/relation_id_to_prompt_dict.txt'
            write_dict(out_file_path, self.relation_dict)
            out_file_path = f'{out_dir}/target_true_id_to_str_dict.txt'
            write_dict(out_file_path, self.target_true_dict)
            out_file_path = f'{out_dir}/target_new_id_to_str_dict.txt'
            write_dict(out_file_path, self.target_new_dict)
            out_file_path = f'{out_dir}/target_new_id_to_relation_id_dict.txt'
            write_dict(out_file_path, self.target_new_relation_dict)

    
    def make_dicts_from_datas(self, datas, out_dir, is_data_list):
        for data in datas:
            data_requesteds = data['requested_rewrite']

            if not is_data_list:
                data_requesteds = [data_requesteds]
            
            for data_requested in data_requesteds:
                prompt = data_requested['prompt']
                relation_id = data_requested['relation_id']

                target_true = data_requested['target_true']['str']
                target_true_id = data_requested['target_true']['id']

                target_new = data_requested['target_new']['str']
                target_new_id = data_requested['target_new']['id']

                add_dict(self.relation_dict, relation_id, prompt)
                add_dict(self.target_true_dict, target_true_id, target_true)
                add_dict(self.target_new_dict, target_new_id, target_new)
                add_dict(self.target_new_relation_dict, target_new_id, relation_id)
        
        self._sorted_dicts()
        self._write_dicts(out_dir)
        print(f'### FindWithWiki.make_dicts_from_datas end ###\n')
    

    def find_triple(self, out_file_path, relation_dict=None, delay=0):
        out_file = open_file(out_file_path, mode='w')

        cnt_find_sub_dict, cnt_find_rel_dict = {}, {}
        cnt, cnt_write = 0, 0

        for i, target_new_id in enumerate(self.target_new_relation_dict.keys()):
            target_new = self.target_new_dict[target_new_id]
            
            if relation_dict is None:
                relation_ids = self.target_new_relation_dict[target_new_id]
            else:
                relation_ids = relation_dict.keys()

            for relation_id in relation_ids:
                find_subject_list, find_relation_list = find_triple_with_wiki(relation_id, target_new_id, 10)

                if find_subject_list is None:
                    while 1:
                        print(f'FindWithWiki.find_triple() [{i}]: [{relation_id}]-[{target_new_id}] {target_new} delay : {delay}s')
                        time.sleep(delay)
                        
                        find_subject_list, find_relation_list = find_triple_with_wiki(relation_id, target_new_id, 10)
                        if find_subject_list is not None:
                            break

                find_subject_list = sorted(list(set(find_subject_list)))
                find_relation_list = sorted(list(set(find_relation_list)))
                add_dict_freq(cnt_find_sub_dict, str(len(find_subject_list)))
                add_dict_freq(cnt_find_rel_dict, str(len(find_relation_list)))

                # 현재 relation 에 대응된 prompt 목록
                if relation_dict is None:
                    prompt_list = self.relation_dict[relation_id]
                else:
                    prompt_list = relation_dict[relation_id]

                for find_subject in find_subject_list:
                    for prompt in prompt_list:
                        out_list = [target_new_id, relation_id, prompt, find_subject, ', '.join(find_relation_list), target_new[0]]
                        out_str = '\t'.join(out_list)

                        out_file.write(f'{out_str}\n')
                        cnt_write += 1
                
                cnt += 1
                if (cnt % 100) == 0:
                    print(f'FindWithWiki.find_triple() [{i}]: {target_new} {cnt} find complet, cnt_write : {cnt_write}')
                out_file.flush()
            if (cnt % 100) != 0:
                print(f'FindWithWiki.find_triple() [{i}]: {target_new} {cnt} find complet, cnt_write : {cnt_write}')
            
            # if delay > 0:
            #     print(f'FindWithWiki.find_triple() [{i}]: {target_new} delay : {delay}s')
            #     time.sleep(delay)
        
        print(f'FindWithWiki.find_triple() {cnt} find complet, cnt_write : {cnt_write}\n')
        out_file.close()

        print(f'FindWithWiki.find_triple() cnt_find_sub_dict : {sorted_dict_key(cnt_find_sub_dict)}')
        print(f'FindWithWiki.find_triple() cnt_find_rel_dict : {sorted_dict_key(cnt_find_rel_dict)}\n')


    def find_knowledge_in_model(self, in_file_path, out_file_path_temp: str, gen_num=-1):
        in_file = open_file(in_file_path, mode='r')
        datas = {}

        while 1:
            line = in_file.readline()
            if not line:
                break

            line = line.strip()
            temp = line.split('\t')
            target_new_id = temp[0].strip()

            if not target_new_id in datas.keys():
                datas[target_new_id] = []
            
            datas[target_new_id].append(line)

        in_file.close()
        print(f'FindWithWiki.find_knowledge_in_model() target_new_ids size : {len(datas)}')

        out_file = open_file(out_file_path_temp.format('', 'txt'), mode='w')
        err_file = open_file(out_file_path_temp.format('_err', 'txt'), mode='w')

        # check_target_new_ids = ['Q462471']
        check_target_new_ids = datas.keys()
        print(f'check_target_new_ids size : {len(check_target_new_ids)}\n')

        results = {}
        cnt_all, cnt_find_all = 0, 0

        for check_target_new_id in check_target_new_ids:
            cnt, cnt_find = 0, 0
            target_news = set()

            for line in datas[check_target_new_id]:
                cnt += 1
                cnt_all += 1
                temp = line.split('\t')

                target_new_id = temp[0].strip()
                relation_id = temp[1].strip()
                prompt = temp[2].strip()
                find_subject = temp[3].strip()
                find_relation = temp[4].strip()
                target_new = temp[5].strip()
                target_news.add(target_new)

                prompt_for_gen = prompt.format(find_subject)

                if gen_num < 1:
                    max_gen_len = len(self._tok(prompt_for_gen)[0]) + len(self._tok(target_new)[0])
                else:
                    max_gen_len = len(self._tok(prompt_for_gen)[0]) + gen_num

                text_gen = self._generate([prompt_for_gen], 1, max_gen_len)
                text_gen_rm_prompt = rm_prompt(prompt_for_gen, text_gen)

                # prompt 바로 뒤에, target_new 가 generation 된 경우만 확인
                if check_start_target(target_new, text_gen_rm_prompt):
                    out_file.write(f'{line}\t{text_gen}\n')
                    out_file.flush()
                    cnt_find += 1
                    cnt_find_all += 1
                else:
                    err_file.write(f'target_new_id : {target_new_id}\n')
                    err_file.write(f'relation_id : {relation_id}\n')
                    err_file.write(f'prompt : {prompt}\n')
                    err_file.write(f'find_subject : {find_subject}\n')
                    err_file.write(f'find_relation : {find_relation}\n')
                    err_file.write(f'prompt_for_gen : {prompt_for_gen}\n')
                    err_file.write(f'text_gen : {text_gen}\n')
                    err_file.write(f'text_gen_rm_prompt : {text_gen_rm_prompt}\n')
                    err_file.write(f'target_new : {target_new}\n\n')
                    err_file.flush()
                
                if (cnt_all % 500) == 0:
                    print(f'{cnt_all} generation, find knowledge in model : {cnt_find_all}')
            if (cnt_all % 500) != 0:
                print(f'{cnt_all} generation, find knowledge in model : {cnt_find_all}')
            
            results[check_target_new_id] = [cnt, cnt_find]

            target_news = ' #%# '.join(sorted(list(target_news)))
            print(f'# {check_target_new_id} : {results[check_target_new_id]}\t\t\t\t\t= {target_news}\n')
        
        out_file.close()
        err_file.close()
        print(f'[ALL] {cnt_all} generation, find_all : {cnt_find_all}\n')

        cnt_file = open_file(out_file_path_temp.format('_cnt', 'txt'), mode='w')
        for key, value in results.items():
            cnt_file.write(f'{key}\t{value[0]}\t{value[1]}\n')
        cnt_file.close()
        print(f'### FindWithWiki.find_knowledge_in_model() end ###\n')





def run(model_name, out_dir, datas=None, select_ids=[], delay=0):
    alg_name = 'ROME'
    hparams_fname = f'{model_name}.json'
    ds_name = 'cf'
    num_edits = 1

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
    finder = FindWithWiki(model_editor._model, model_editor._tok)

    if datas is None:
        model_editor.load_data()
        datas = model_editor._ds
        is_mquake = False
    else:
        is_mquake = True
    
    finder.make_dicts_from_datas(datas, out_dir, is_mquake)

    out_file_path = f'{out_dir}/find_triple_with_wiki.txt'
    
    if not is_mquake:
        finder.find_triple(out_file_path, delay=delay)
    else:
        finder.find_triple(out_file_path, RELATION_DICT_MQUAKE, delay)
    
    in_file_path = out_file_path
    out_file_path_temp = f'{out_dir}/find_knowledge_in_model_with_wiki' + '{}' + '.{}'
    finder.find_knowledge_in_model(in_file_path, out_file_path_temp)

    in_file_path = out_file_path_temp.format('', 'txt')
    out_file_path = out_file_path_temp.format('', 'json')
    convert_to_json(in_file_path, out_file_path, set(select_ids))


if __name__ == "__main__":
    home_dir = '/home/nlpshlee/dev_env/git/repos/memit'

    in_file = open_file(f'{home_dir}/data/MQuAKE-CF-2k-v2.json', mode='r')
    mquake_datas = json.load(in_file)

    # 1. counterfact, gpt2-xl
    model_name = 'gpt2-xl'
    out_dir = f'{home_dir}/data/find_with_wiki_from_cf_gpt'
    # run(model_name, out_dir, delay=60)

    # 2. mquake, gpt2-xl
    model_name = 'gpt2-xl'
    out_dir = f'{home_dir}/data/find_with_wiki_from_mquake_gpt'
    # run(model_name, out_dir, mquake_datas, delay=60)

    # 3. mquake, llama3-8B
    model_name = 'meta-llama/Meta-Llama-3-8B'
    out_dir = f'{home_dir}/data/find_with_wiki_from_mquake_llama'
    # run(model_name, out_dir, mquake_datas, delay=60)

