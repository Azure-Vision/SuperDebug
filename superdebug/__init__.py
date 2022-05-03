# coding=utf-8
# from superdebug import debug, mail
import torch
try:
    from torchvision.utils import save_image
except:
    save_image is None
tf = None
if False:
    try:
        import tensorflow as tf
    except:
        pass
import numpy as np
import sys
import os
import time
import re
from collections import OrderedDict, defaultdict
try:
    from PIL import Image
except:
    Image = None

# 开关 #################################
ON_DEBUG = True # debug总开关
PLAIN = False  # 开启则仅普通的打印（至终端或debug.log）
MAX_LOG = -1  # 0: 不debug, -1: 无限输出 NOTE: 无输出？可能这里设成了0，或者数量不够高、没到需要输出的变量！
FULL = False  # 是否输出完整的tensor、string内容，而不用...进行省略
TO_FILE = False  # 是否写入debug.log
PRINT = True  # 是否打印至终端
BUGGY = True  # 便捷地debug（出现bug则进入自动进入调试模式）
PEEK_LAYER = 3  # 详细打印至第几层，不详细打印可使用0，详细打印建议用3
MAX_PEEK_ITEM = 3 # 详细打印几项，标准为2
MAX_STR_LEN = 540 # 最长打印的字符串长度，推荐： 540 ，无限大： 9999999999999999
SAVE_IMAGE_NORM = False # 把tensor保存成图片时是否normalize
# 控制是否打印细节：debug(True/False, xxx, xxx)，False则只打印形状

# 教程 #################################
# 功能1: debug(xxx) : 用黄色字体打印出xxx的形状及具体值，debug(False, xxx)则只打印形状，不打印具体值。更多控制开关见上方。
# 功能2: mark(xxx) : 标记运行到了某个位置，若有输入，则用黄色字体打印出xxx值，若仅用mark()无输入，则打印mark()所在的位置
# 功能3: 在出错时跳至ipdb界面，便捷debug

# 实现 #################################
try:
    MY_QQ_EMAIL = os.environ["MY_QQ_EMAIL"] # Email address
    MY_QQ_EMAIL_PWD = os.environ["MY_QQ_EMAIL_PWD"] # Password
except:
    print("为了使用邮件提醒功能，请设置环境变量MY_QQ_EMAIL（QQ邮箱地址）与MY_QQ_EMAIL_PWD（QQ邮箱授权码）")

debug_count = 1
debug_file = None
debug_path = "super_debug"
if os.path.exists(debug_path):
    os.system("rm -r " + debug_path)
os.makedirs(debug_path, exist_ok=True)
log_path = os.path.join(debug_path, "debug.log")
os.system("touch " + log_path)
image_count = {}
simple_types = [str, int, float, bool]


class ExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if not BUGGY:
            return
        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode='Plain',
                                                color_scheme='Linux', call_pdb=1)
        return self.instance(*args, **kwargs)


sys.excepthook = ExceptionHook()


def get_pos(level=1, end="\n"):
    position = """{}:{} {}""".format(
    # position = """"{}", line {}, in {}""".format(
        sys._getframe(level).f_code.co_filename,  # 当前文件名
        sys._getframe(level).f_lineno,  # 当前行号
        sys._getframe(level).f_code.co_name,  # 当前函数/module名
    )
    return position

def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def print_yellow(text, end="\n"):
    print(f"\033[1;33m{text}\033[0m", end=end)


def normalize(tensor):
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    tensor = (tensor - min_value) / (max_value - min_value)
    return tensor

def print_image(tensor, name, is_np = False):
    if name not in image_count:
        image_count[name] = 0
    file_path = os.path.join(debug_path, f"tensor_{debug_count}_{name}_{image_count[name]}.jpg")
    normallized_file_path = os.path.join(debug_path, f"tensor_{debug_count}_{name}_{image_count[name]}_norm.jpg")
    image_count[name] += 1
    if Image is not None and type(tensor) == Image.Image:
        tensor.save(file_path)
    else:
        if is_np:
            tensor = torch.Tensor(tensor)
        normalized_tensor = normalize(tensor)
        try:
            if save_image is not None:
                if SAVE_IMAGE_NORM:
                    save_image(normalized_tensor, normallized_file_path)
                else:
                    save_image(tensor, file_path)
        except Exception:
            pass
def mark(marker=None):
    print("Mark is deprecated. Use debug() instead.")


def logging(*message, end="\n"):
    """同时输出到终端和debug.log"""
    message = " ".join([str(_) for _ in message])
    if PRINT:
        print_yellow(message, end=end)
    if debug_file and not debug_file.closed:
        message = re.sub("\033\[.*?m", "", message)
        debug_file.write(message + end)
logging("------------------\033[0m\033[1;31m", get_time(), "\033[0m\033[1;33m------------------")


def info(var, name="?", detail=True, layer=0):
    """递归打印变量"""
    space = "   "
    if type(var) == int or type(var) == float:
        logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "num val:", var)
    else:
        if var is None:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "None")
        elif type(var) == str:
            length = len(var)
            if not FULL and len(var) >= MAX_STR_LEN:
                var = var[:MAX_STR_LEN - 20] + " ... " + var[-20:]
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "str len", str(length)+":", var)
        elif type(var) == bool:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "bool:", var)
        elif type(var) == list:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "list size:", len(var), end="")
            if layer < PEEK_LAYER and len(var) > 0 and type(var[0]) not in simple_types: # a list of complex variables
                logging(f" [{min(len(var), 3)*'.'}]")
                for no, item in enumerate(var[:MAX_PEEK_ITEM]):
                    info(item, "item " + str(no) + ": ", detail, layer + 1)
                if len(var) > MAX_PEEK_ITEM:
                    logging(space * (layer + 1), len(var) - MAX_PEEK_ITEM, "extra items")
            else:
                var_str = str(var)
                if len(var) > 0 and type(var[0]) in simple_types and all([type(var[i]) == type(var[0]) for i in range(len(var))]): # a list of variables of the same simple type
                    if len(var_str) >= MAX_STR_LEN + 3: # too long
                        # show_num = len(var_str[:MAX_STR_LEN].split(","))
                        logging(" val:", f"{var_str[:MAX_STR_LEN]} ... and extra items]" if detail else "*")
                    else:
                        logging(" val:", var_str if detail else "*")
                elif layer < PEEK_LAYER: # variables of different simple types
                    logging(f" [{min(len(var), 3)*'.'}]")
                    for no, item in enumerate(var):
                        info(item, "item " + str(no) + ": ", detail, layer + 1)
                else:
                    logging(" val:", var_str if detail else "*")
        elif type(var) == tuple:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "tuple size:", len(var), end="")
            if layer < PEEK_LAYER and len(var) > 0:
                logging(f" ({min(len(var), 3)*'.'})")
                for no, item in enumerate(var):
                    info(item, "item " + str(no) + ": ", detail, layer + 1)
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == set:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "set size:", len(var), end="")
            if layer < PEEK_LAYER and len(var) > 0:
                logging(" {" + min(len(var), 3)*'.' + "}")
                for no, item in enumerate(var):
                    info(item, "item " + str(no) + ": ", detail, layer + 1)
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == dict:
            dict_keys = sorted(list(var.keys()))
            dict_end = ""
            if len(dict_keys) >= 100:
                dict_keys = dict_keys[:100]
                dict_end = "... and extra items]"
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "dict {" + min(len(var), 3)*'.' + "} " + f"with {len(dict_keys)} keys", dict_keys, end=dict_end)
            if layer < PEEK_LAYER and len(var) > 0:
                logging("")
                for key in dict_keys:
                    info(var[key], key, detail, layer + 1)
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == OrderedDict:
            dict_keys = sorted(list(var.keys()))
            dict_end = ""
            if len(dict_keys) >= 100:
                dict_keys = dict_keys[:100]
                dict_end = "... and extra items]"
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "OrderedDict {" + min(len(var), 3)*'.' + "} " + f"with {len(dict_keys)} keys", dict_keys, end=dict_end)
            if layer < PEEK_LAYER and len(var) > 0:
                logging("")
                for key in dict_keys:
                    info(var[key], key, detail, layer + 1)
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == defaultdict:
            tmp_val = 12341231354124
            assert tmp_val not in var
            default_val = var[tmp_val]
            del var[tmp_val]

            dict_keys = sorted(list(var.keys()))
            dict_end = ""
            if len(dict_keys) >= 100:
                dict_keys = dict_keys[:100]
                dict_end = "... and extra items]"

            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "defaultdict {" + min(len(var), 3)*'.' + "} with default", default_val, f"{len(dict_keys)} keys", dict_keys, end=dict_end)
            if layer < PEEK_LAYER and len(var) > 0:
                logging("")
                for key in dict_keys:
                    info(var[key], key, detail, layer + 1)
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == torch.Tensor:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "Tensor size:", var.shape, "val:", var if detail else "*")
            print_image(var, name)
        elif type(var) == np.ndarray:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "ndarray size:", var.shape,
                    "val:", var if detail else "*")
            print_image(var, name, True)
        elif tf is not None and type(var) == tf.Tensor:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "Tensor size:", var.shape, "val:", var if detail else "*")
        elif Image is not None and type(var) == Image.Image:
            print_image(var, name)
        else:
            var_type = str(type(var)).split("'")[1]
            try:
                if layer >= PEEK_LAYER:
                    raise Exception("Too many layers")
                props = var.__dict__
                logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", var_type, "with props", list(props.keys()), end="")
                prop_valid = False
                for key in props:
                    if not key.startswith("_"):
                        if not prop_valid:
                            prop_valid = True
                            logging("")
                        info(props[key], key, detail, layer + 1)
                if not prop_valid:
                    logging(" val:", var)
            except Exception:
                var_type = str(type(var))[8:-2]
                logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", var_type, "with val: ", var)


def debug(*args, **kwargs):
    """debug打印主入口"""
    global ON_DEBUG
    global debug_count
    global debug_file
    global TO_FILE
    global PLAIN
    if not ON_DEBUG:
        return 
    if TO_FILE:
        try:
            debug_file = open(log_path, "a", encoding='utf-8')
        except:
            debug_file = None
    logging("------------------\033[0m\033[1;31m", get_time(), "\033[0m\033[1;33m------------------")
    if PLAIN:
        logging(*args, **kwargs, end="\n")
        if TO_FILE and debug_file is not None:
            debug_file.close()
        return
    global FULL
    if FULL:
        torch.set_printoptions(profile="full")
        np.set_printoptions(threshold=sys.maxsize)
    count = 0
    if MAX_LOG != -1 and debug_count >= MAX_LOG:
        if debug_file:
            debug_file.close()
        return
    detail = True
    if args and type(args[0]) is bool:
        detail = args[0]
        args = args[1:]
    keys = list(kwargs.keys())
    if len(args) + len(kwargs) == 0:
        logging(f"\033[0m\033[1;32mMARK:\033[0m\033[1;33m at \033[0m\033[1;32m{get_pos(level=2)}\033[0m\033[1;33m")
    elif len(args) == 1 and len(kwargs) == 0 and type(args[0]) == str:
        logging(f"\033[0m\033[1;32mDEBUG:\033[0m\033[1;33m at \033[0m\033[1;32m{get_pos(level=2)}\033[0m\033[1;33m")
        logging(f"\033[0m\033[1;36m{args[0]}\033[0m\033[1;33m")
    else:
        logging(f"\033[0m\033[1;32mDEBUG:\033[0m\033[1;33m {len(args) + len(kwargs)} vars: {['?' for _ in args] + keys}, at \033[0m\033[1;32m{get_pos(level=2)}\033[0m\033[1;33m")
        for var in args:
            logging(f"{count} / {debug_count}.",  end=" ")
            info(var, detail=detail)
            debug_count += 1
            count += 1
        for key in keys:
            logging(f"{count} / {debug_count}.", end=" ")
            info(kwargs[key], key, detail=detail)
            count += 1
            debug_count += 1
    logging("------------------\033[0m\033[1;31m", get_time(), "\033[0m\033[1;33m------------------")
    if TO_FILE and debug_file is not None:
        debug_file.close()

def mail(subject = "Progress Notification", message = ""):
    from email.mime.text import MIMEText
    subject = f"[SUPERDEBUG] {subject}"
    message = f"{message}\nThis email is sent at   {get_pos(level=2)}"
    mail = MIMEText(message)
    mail['Subject'] = subject
    mail['From'] = MY_QQ_EMAIL
    mail['To'] = MY_QQ_EMAIL

    import smtplib
    smtp=smtplib.SMTP()
    smtp.connect('smtp.qq.com', 25)
    smtp.login(MY_QQ_EMAIL, MY_QQ_EMAIL_PWD)

    smtp.sendmail(MY_QQ_EMAIL, MY_QQ_EMAIL, mail.as_string()) # To是接收邮箱
