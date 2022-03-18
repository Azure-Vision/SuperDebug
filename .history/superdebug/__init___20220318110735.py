# from debug import debug, mark
import torch
try:
    from torchvision.utils import save_image
except:
    pass
import numpy as np
import sys
import os
try:
    from PIL import Image
except:
    pass

# 开关 #################################
ON_DEBUG = True # debug总开关
PLAIN = False  # 开启则仅普通的打印（至终端或debug.log）
MAX_LOG = -1  # 0: 不debug, -1: 无限输出 NOTE: 无输出？可能这里设成了0，或者数量不够高、没到需要输出的变量！
FULL = False  # 是否输出完整的tensor内容，而不用...进行省略
TO_FILE = True  # 是否写入debug.log
PRINT = True  # 是否打印至终端
BUGGY = True  # 便捷地debug（出现bug则进入自动进入调试模式）
PEEK_LAYER = 0  # 详细打印至第几层，标准为0，建议用3
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
    position = """"{}", line {}, in {}""".format(
        sys._getframe(level).f_code.co_filename,  # 当前文件名
        sys._getframe(level).f_lineno,  # 当前行号
        sys._getframe(level).f_code.co_name,  # 当前函数/module名
    )
    return position


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
    if type(tensor) == Image.Image:
        tensor.save(file_path)
    else:
        if is_np:
            tensor = torch.Tensor(tensor)
        normalized_tensor = normalize(tensor)
        try:
            if SAVE_IMAGE_NORM:
                save_image(normalized_tensor, normallized_file_path)
            else:
                save_image(tensor, file_path)
        except Exception:
            pass
def mark(marker=None):
    if marker is not None:
        print_yellow(marker)
    else:
        print_yellow(get_pos(level=2))


def logging(*message, end="\n"):
    """同时输出到终端和debug.log"""
    message = " ".join([str(_) for _ in message])
    if debug_file:
        debug_file.write(message + end)
    if PRINT:
        print_yellow(message, end=end)


def info(i, name="", detail=True, layer=0):
    """递归打印变量"""
    global PEEK_LAYER
    sep = "   "
    if type(i) == int or type(i) == float:
        logging(sep * layer, name, "num val:", i)
    else:
        if type(i) == str:
            logging(sep * layer, name, "str:", i)
        elif type(i) == bool:
            logging(sep * layer, name, "bool:", i)

        elif type(i) == list:
            logging(sep * layer, name, "list size:", len(i), end="")
            if layer < PEEK_LAYER and len(i) > 0:
                logging("")
                info(i[0], "0th item:", detail, layer + 1)
            else:
                logging(" val:", i if detail else "*")
        elif type(i) == dict:
            logging(sep * layer, name, "dict with keys", list(i.keys()))
            for key in i:
                info(i[key], key, detail, layer + 1)
        elif type(i) == tuple:
            logging(sep * layer, name, "tuple size:", len(i), "")
            if layer < PEEK_LAYER and len(i) > 0:
                for no, item in enumerate(i):
                    info(item, str(no) + ". ", detail, layer + 1)
            else:
                logging(" val:", i if detail else "*")
        elif type(i) == torch.Tensor:
            logging(sep * layer, name, "Tensor size:", i.shape,
                    "val:", i if detail else "*")
            print_image(i, name)
            
        elif type(i) == np.ndarray:
            logging(sep * layer, name, "ndarray size:", i.shape,
                    "val:", i if detail else "*")
            print_image(i, name, True)
        elif type(i) == Image.Image:
            print_image(i, name)
        else:
            try:
                j = float(i)
            except Exception:
                logging(sep * layer, name, str(type(i)) + " with val: ", i)
            else:
                logging(sep * layer, name, "num val:", j, type(i))


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
    logging(
        f"DEBUG: {len(args) + len(kwargs)} vars: {['?' for _ in args] + keys}, at {get_pos(level=2)}")
    for i in args:
        logging(f"{count} / {debug_count}.",  end=" ")
        info(i, detail=detail)
        debug_count += 1
        count += 1
    for i in keys:
        logging(f"{count} / {debug_count}.", end=" ")
        info(kwargs[i], i, detail=detail)
        count += 1
        debug_count += 1
    logging("-------------------------------------")
    if TO_FILE:
        debug_file.close()