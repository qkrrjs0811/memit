import os, sys, json, string, re
from itertools import islice

ENCODING = "UTF-8"
DELIM_KEY = "\t"
FILE_EXT = "."

# 모든 일반적인 기호 포함
PUNCTUATION = string.punctuation

class TXT_OPTION:
    OFF = 0
    UPPER = 1
    LOWER = 2
    RM_SPACE = 4


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


def exists(file_path: str):
    if file_path == None or len(file_path) == 0:
        return False

    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True

    return False


def json_file_to_dict(in_file_path: str, encoding=ENCODING):
    try:
        if exists(in_file_path):
            file = open_file(in_file_path, encoding, 'r')

            # 파일을 읽을 때는, load() 호출
            json_dict = json.load(file)

            return json_dict

    except Exception as e:
        logging_error("json_file_to_dict()", e)
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
