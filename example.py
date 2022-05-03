from collections import OrderedDict, defaultdict
from superdebug import debug
import time

debug("程序开始运行了，记得看时间哦")

story = ["从前有座山", "山里有座庙", "庙里有个老和尚", "老和尚在给小和尚讲故事", "讲什么呢"] * 10000 # 重复10000遍

config = OrderedDict({"标题": "和尚的故事", "内容": story, "更多信息": {"作者": ("老和尚", "小和尚", "小小和尚"), "时间": {"创作时间": "很久很久以前", "timestamp": time.time()}}})
debug("你可以同时debug多个变量，没有被设置名字的变量会显示为'?'", story, 提示 = "这个故事实在太长了，我们只打印其中的一段", config = config, 另一个提示 = "我们可以清晰地看出变量的嵌套结构")

from superdebug import  mail
debug("让我们模拟一段耗时的任务，任务结束后我会向你发送邮件")
time.sleep(10)
mail("任务已处理完成")
debug("你收到邮件了吗？")

raise Exception("哎呀，程序出错了... 去在ipdb界面中调试程序吧")