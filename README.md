# SuperDebug

## 使用方法：

准备工作：在需要工程目录下添加debug.py，在其他文件from debug import debug, mark

### 功能1: `debug(var)`

用黄色字体打印出`var`所在位置、数组（list、numpy的array、torch的tensor）形状及具体值

进阶功能:

1. 一次性打印多个变量：`debug(var1, var2, var3, ...)`

2. 同步输出在终端（`PRINT = True`）和debug.log文件中（`TO_FILE = True`）

3. 对于一些循环程序，你或许只想打印有限次，只需设置 `MAX_LOG = 想打印的次数` 即可

4. 如果你希望只打印变量的形状，而不打印变量的内容，可以使用`debug(False, var)`

5. 打印完整矩阵内容，禁止pytorch、numpy自动省略：`FULL = True`

6. 对于一些套娃list，可以控制详细打印至第几层，标准为0，建议用3： `PEEK_LAYER = 0`


### 功能2: `mark(var)`

标记运行到了某个位置，若有输入，则用黄色字体打印出 `var` 的值，若仅用`mark()`无输入，则打印`mark()`所在的位置

### 功能3: 在出错时跳至ipdb调试界面，便捷debug

自动停在出错的那一步（需设置`BUGGY = True`）：当出现Exception，程序不会退出，而是会暂停在出现Exception的那一步。

这时你可以使用`p var`（输出var的值）, `up`（上一步）, `down`（下一步）等控制命令进行调试。


