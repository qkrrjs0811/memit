import os, sys, json, string, re
from itertools import islice


ENCODING = "UTF-8"
DELIM_KEY = "\t"
FILE_EXT = "."


# 모든 일반적인 기호 포함
PUNCTUATION = string.punctuation


class LOG_OPTION:
    STDOUT = 1
    STDERR = 2


def check_option(option1: int, option2: int):
    if option1 == option2:
        return True
    elif (option1 & option2) != 0:
        return True
    else:
        return False


def logging(msg: str, option=LOG_OPTION.STDOUT):
    if check_option(option, LOG_OPTION.STDOUT):
        print(msg)
    if check_option(option, LOG_OPTION.STDERR):
        print(msg, file=sys.stderr)


def logging_error(call_path: str, e: Exception):
    logging(f"### (ERROR) {call_path} error : {e}\n", LOG_OPTION.STDERR)


def json_str_to_dict(json_str: str):
    try:
        # 문자열을 읽을 때는, loads() 호출
        return json.loads(json_str)

    except Exception as e:
        logging_error("json_str_to_dict()", e)
        return None


def to_json_str(input, indent=4):
    try:
        return json.dumps(input, ensure_ascii=False, indent=indent)

    except Exception as e:
        logging_error("to_json_str()", e)
        return ""


def write_json_to_file(json_dict, out_file_path, encoding=ENCODING):
    out_file = open_file(out_file_path, encoding, 'w')
    out_file.write(to_json_str(json_dict))
    out_file.close()


def is_empty(text: str, trim_flag=True):
    if text is None:
        return True
    
    if trim_flag:
        text = text.strip()
    
    if len(text) == 0:
        return True
    
    return False


def is_symbol(text: str, symbols=PUNCTUATION):
    if is_empty(text):
        return False

    for c in text:
        if c == ' ' or c == '\t' or c == '\n':
            continue
        if not c in symbols:
            return False

    return True


def contains_symbol(text: str, symbols=PUNCTUATION):
    if is_empty(text):
        return False

    for c in text:
        if c in symbols:
            return True

    return False


def remove_delim_multi(text: str):
    return re.sub(r'[\t\n ]+', ' ', text).strip()


def remove_symbol_edge(text: str, symbols=PUNCTUATION):
    return text.strip(symbols)


def get_file_name(file_path: str, rm_ext_flag=False) :
    file_name = os.path.basename(file_path)

    if rm_ext_flag :
        idx = file_name.rfind(FILE_EXT)

        if idx != -1 :
            file_name = file_name[:idx]
    
    return file_name


def get_file_paths(in_path: str, inner_flag=True) :
    file_paths = []

    if inner_flag :
        for (parent_path, dirs, file_names) in os.walk(in_path) :
                for file_name in file_names :
                    file_path = os.path.join(parent_path, file_name)

                    if os.path.isfile(file_path) :
                        file_paths.append(file_path)
    else :
        file_names = os.listdir(in_path)
        for file_name in file_names :
            file_path = os.path.join(in_path, file_name)

            if os.path.isfile(file_path) :
                file_paths.append(file_path)

    return file_paths


def exists(file_path: str):
    if file_path == None or len(file_path) == 0:
        return False

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True

    return False


def load_json_file(in_file_path: str, encoding=ENCODING, do_print=False):
    try:
        if exists(in_file_path):
            file = open_file(in_file_path, encoding, 'r')

            # 파일을 읽을 때는, load() 호출
            datas = json.load(file)

            if do_print:
                print(f'# falcon_util.load_json_file() datas size : {len(datas)}, in_file_path : {in_file_path}')

            return datas

    except Exception as e:
        logging_error("# falcon_util.load_json_file() error : ", e)
        return None

    return None


def make_parent(file_path: str):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)


def open_file(file_path: str, encoding=ENCODING, mode='r'):
    if mode.find('w') != -1 or mode.find('a') != -1:
        make_parent(file_path)

    if is_empty(encoding, True) or mode.find('b') != -1:
        return open(file_path, mode)
    else:
        return open(file_path, mode, encoding=encoding)


def file_to_freq_dict(in_file_path: str, encoding=ENCODING, delim_key=DELIM_KEY, in_dict=dict()):
    in_file = open_file(in_file_path, encoding, mode='r')

    while 1:
        line = in_file.readline()
        if not line:
            break

        temp = line.strip().split(delim_key)
        key = temp[0].strip()
        value = int(temp[1].strip())

        add_dict_freq(in_dict, key, value)
    in_file.close()

    return in_dict


def window(seq, n=2):
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i: i + n]


def is_true(in_list: list):
    for temp in in_list:
        if not bool(temp):
            return False
    
    return True


'''
    key를 기준으로 정렬
        - is_reverse = False : 오름 차순
        - is_reverse = True : 내림 차순
'''
def sorted_dict_key(in_dict: dict, is_reverse=False):
    return dict(sorted(in_dict.items(), key=lambda item:item[0], reverse=is_reverse))

'''
    value를 기준으로 정렬
        - is_reverse = False : 오름 차순
        - is_reverse = True : 내림 차순
'''
def sorted_dict_value(in_dict: dict, is_reverse=False):
    return dict(sorted(in_dict.items(), key=lambda item:item[1], reverse=is_reverse))

'''
    key를 기준으로 오름 차순 정렬, value를 기준으로 내림 차순 정렬
'''
def sorted_dict(in_dict: dict):
    return sorted_dict_value(sorted_dict_key(in_dict, False), True)


def add_dict_freq(in_dict: dict, key, value=1):
    if key in in_dict.keys():
        in_dict[key] += value
    else:
        in_dict[key] = value


def write_dict_freq(out_dict: dict, out_file_path: str, encoding=ENCODING, delim=DELIM_KEY) :
    file = open_file(out_file_path, encoding, 'w')

    items = out_dict.items()
    for item in items :
        file.write(f"{item[0]}{delim}{item[1]}\n")
    
    file.close()


def trim(input_list: list, rm_empty_flag: bool):
    if not rm_empty_flag:
        for i in range(len(input_list)):
            if input_list[i] is None:
                input_list[i] = ""
            else :
                input_list[i] = str(input_list[i]).strip()
    else:
        result = []

        for i in range(len(input_list)):
            if not is_empty(input_list[i], True):
                result.append(str(input_list[i]).strip())
        
        return result
