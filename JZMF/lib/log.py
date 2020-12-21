# -*- coding: utf-8 -*-
import logging
import os
import time

# #日志名称和日志路径
_LogTime = time.strftime('%Y%m%d', time.localtime(time.time()))
log_path = os.path.join('.', 'logs')
if os.path.isdir(log_path):
   pass
else:
   os.mkdir(log_path)
logfile = log_path + "/logs" + _LogTime + '.txt'

###############################   logging 基础设置   ################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    filename=logfile,
                    filemode='a')

#定义一个StreamHandler，将INFO级别或更高的日志信息打印到标准错误，并将其添加到当前的日志处理对象#
console = logging.StreamHandler()
console.setLevel(logging.INFO) # DEBUG INFO WARNING ERROR
formatter = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
console.setFormatter(formatter)

logging.getLogger('').addHandler(console)


if __name__ == '__main__':
    logging.info("Start print log")
    logging.debug("Do something")
    logging.debug("Do something")
    logging.warning("Something maybe fail.")
    try:
        open("sklearn.txt", "rb")
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        logging.error(e)