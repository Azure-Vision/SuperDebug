# SuperDebug

## 使用方法：

`pip install superdebug`

`from superdebug import debug, mail`

### 功能1: `debug(var)`

用黄色字体打印出（与普通的打印区分）：

- `var`所在具体行数（command / ctrl + 单击 即可跳转到文件相应位置）
- 数组形状（限list、numpy的array、torch的tensor）
- `var`的具体值

进阶功能:

1. 一次性打印多个变量：`debug(var1, var2, var3, ...)`
2. 同步输出在终端（`PRINT = True`）和同目录下的debug.log文件中（`TO_FILE = True`）
3. 对于一些循环程序，你或许只想打印有限次，只需设置 `MAX_LOG = 想打印的次数` 即可
4. 如果你希望只打印变量的形状，而不打印变量的内容，可以使用 `debug(False, var)`
5. 打印完整矩阵内容，禁止pytorch、numpy自动省略：`FULL = True`
6. 对于一些套娃list / dict，可以控制详细打印至第几层，标准为0，建议用3： `PEEK_LAYER = 0`

### 功能2: `debug()`

简单标记程序已经运行到了某一位置，可用于判断程序在哪一处卡在、程序进入了哪一个if分支等，也可以查看程序运行到这一位置的时间。

### 功能3: 在出错时跳至ipdb调试界面，便捷debug

自动停在出错的那一步（需设置 `BUGGY = True`）：当出现Exception，程序不会退出，而是会暂停在出现Exception的那一步。

这时你可以使用 `var`（输出var的值）, `up`（上一步）, `down`（下一步）等控制命令进行调试。更多命令同ipdb

### 功能4: 在指定位置停下（设置断点）

在想设置断点处写 `raise Exception()`，与功能3同理，程序将会停在exception处。

### 功能5: 邮件提醒

用于在某一任务完成后发送邮件通知。

需配置环境变量 `MY_QQ_EMAIL`（QQ邮箱地址）与 `MY_QQ_EMAIL_PWD`（QQ邮箱授权码）。

使用 `mail("邮件主题", "邮件内容")`即可便捷地向你自己的邮箱发送通知，可用于提醒模型已训练完毕、某项耗时的任务已处理完成等。
