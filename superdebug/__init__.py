# from debug import debug, mark
import torch
try:
    from torchvision.utils import save_image
except:
    save_image is None
try:
    import tensorflow as tf
except:
    tf = None
import numpy as np
import sys
import os
import time
try:
    from PIL import Image
except:
    Image = None

# 开关 #################################
ON_DEBUG = True # debug总开关
PLAIN = False  # 开启则仅普通的打印（至终端或debug.log）
MAX_LOG = -1  # 0: 不debug, -1: 无限输出 NOTE: 无输出？可能这里设成了0，或者数量不够高、没到需要输出的变量！
FULL = False  # 是否输出完整的tensor、string内容，而不用...进行省略
TO_FILE = True  # 是否写入debug.log
PRINT = True  # 是否打印至终端
BUGGY = True  # 便捷地debug（出现bug则进入自动进入调试模式）
PEEK_LAYER = 3  # 详细打印至第几层，不详细打印可使用0，详细打印建议用3
MAX_PEEK_ITEM = 2 # 详细打印几项，标准为2
MAX_STR_LEN = 220 # 最长打印的字符串长度
SAVE_IMAGE_NORM = False # 把tensor保存成图片时是否normalize
# 控制是否打印细节：debug(True/False, xxx, xxx)，False则只打印形状

# 教程 #################################
# 功能1: debug(xxx) : 用黄色字体打印出xxx的形状及具体值，debug(False, xxx)则只打印形状，不打印具体值。更多控制开关见上方。
# 功能2: mark(xxx) : 标记运行到了某个位置，若有输入，则用黄色字体打印出xxx值，若仅用mark()无输入，则打印mark()所在的位置
# 功能3: 在出错时跳至ipdb界面，便捷debug

# 实现 #################################

debug_count = 1
debug_file = None
debug_path = "super_debug"
if os.path.exists(debug_path):
    os.system("rm -r " + debug_path)
os.mkdir(debug_path)
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
    if debug_file and not debug_file.closed:
        debug_file.write(message + end)
    if PRINT:
        print_yellow(message, end=end)
logging("------------------\033[0m\033[1;31m", get_time(), "\033[0m\033[1;33m--", end = "")


def info(var, name="?", detail=True, layer=0):
    """递归打印变量"""
    space = "   "
    if type(var) == int or type(var) == float:
        logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "num val:", var)
    else:
        if type(var) == str:
            length = len(var)
            if not FULL and len(var) >= MAX_STR_LEN:
                var = var[:MAX_STR_LEN - 20] + " ... " + var[-20:]
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "str len", str(length)+":", var)
        elif type(var) == bool:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "bool:", var)
        elif type(var) == list:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "list size:", len(var), end="")
            if layer < PEEK_LAYER and len(var) > 0 and type(var[0]) not in simple_types:
                logging("")
                for no, item in enumerate(var[:MAX_PEEK_ITEM]):
                    info(item, "item " + str(no) + ": ", detail, layer + 1)
                if len(var) > MAX_PEEK_ITEM:
                    logging(space * (layer + 1), len(var) - MAX_PEEK_ITEM, "extra items")
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == tuple:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "tuple size:", len(var), "")
            if layer < PEEK_LAYER and len(var) > 0:
                for no, item in enumerate(var):
                    info(item, str(no) + ". ", detail, layer + 1)
            else:
                logging(" val:", var if detail else "*")
        elif type(var) == dict:
            logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "dict with keys", list(var.keys()))
            for key in var:
                info(var[key], key, detail, layer + 1)
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
            try:
                j = float(var)
            except Exception:
                logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", str(type(var)) + " with val: ", var)
            else:
                logging(space * layer, f"\033[0m\033[1;36m{name}\033[0m\033[1;33m", "num val:", j, type(var))


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
        debug_file = open(log_path, "a")
    logging("--\033[0m\033[1;31m", get_time(), "\033[0m\033[1;33m------------------")
    if PLAIN:
        logging(*args, **kwargs, end="\n")
        if TO_FILE:
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
    logging("------------------\033[0m\033[1;31m", get_time(), "\033[0m\033[1;33m--", end = "")
    if TO_FILE:
        debug_file.close()