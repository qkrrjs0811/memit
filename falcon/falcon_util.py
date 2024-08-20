import os, sys, json
from itertools import islice

ENCODING = "UTF-8"
DELIM_KEY = "\t"
FILE_EXT = "."


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

