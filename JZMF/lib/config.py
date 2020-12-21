# -*- coding: utf-8 -*-
# import os
#
# import pandas as pd
# import pymysql
from matplotlib.font_manager import FontProperties

# 贝格数据库(内网)
DBHOST_BG = "192.168.0.238"
DBUSER_BG = "jijing"
DBPWD_BG = "66W7QSEUNKqz829U"
DBNAME_BG = "bigdata"
DBPORT_BG = 3306
DBCHAR_BG = "gbk"

# 聚源数据库(内网)
DBHOST_JY = "192.168.0.238"
DBUSER_JY = "fengpan"
DBPWD_JY = "zugLrLpYjW9u9T2e4w8p"
DBNAME_JY = "gildata"
DBPORT_JY = 3306
DBCHAR_JY = "gbk"

# 经传数据中心
DBHOST_JZ = "139.159.176.118"
DBUSER_JZ = "dcr"
DBPWD_JZ = "acBWtXqmj2cNrHzrWTAciuxLJEreb*4EgK4"
DBNAME_JZ = "datacenter"
DBPORT_JZ = 3306
DBCHAR_JZ = "gbk"

# mongodb
"""
该账户只有读的权限，没有修改的权限
"""
DB_IP = "192.168.0.104"
DB_GUEST_USERNAME = "JZ_fellow"
DB_GUEST_PASSWORD = "JZ_fellow"
DB_PORT = 27017
DB_CHAR = "SCRAM-SHA-1"

# db
DB_STOCK_MIN = "stock_min_data"
DB_STOCK_MIN_CROSS_CODE = "stock_min_cross_code_data"
DB_FUTURE_MIN = "future_min_data"
DB_STATUS = 'status_db'
DB_INDEX_MIN = 'index_min_data'
DB_INDEX_MIN_CROSS_CODE = 'index_min_cross_code_data'
DB_HOLDING = 'holding'

# collection
STOCK_STATUS_COLLECTION = 'stock_min_status'
INDEX_STATUS_COLLECTION = 'index_min_status'
STOCK_HOLDING_COLLECTION = 'stock_holding'


def query_bg(sql):
    """
    查询贝格数据库
    返回：pd.DataFrame
    """

    con = pymysql.connect(host=DBHOST_BG,
                          port=DBPORT_BG,
                          user=DBUSER_BG,
                          password=DBPWD_BG,
                          db=DBNAME_BG,
                          charset=DBCHAR_BG)

    df = pd.read_sql(sql, con)
    con.close()

    return df


def query_jy(sql):
    """
    查询聚源数据库
    返回：pd.DataFrame
    """

    con = pymysql.connect(host=DBHOST_JY,
                          port=DBPORT_JY,
                          user=DBUSER_JY,
                          password=DBPWD_JY,
                          db=DBNAME_JY,
                          charset=DBCHAR_JY)

    df = pd.read_sql(sql, con)
    con.close()

    return df


def query_jz(sql):
    """
    查询经传数据库
    返回：pd.DataFrame
    """

    con = pymysql.connect(host=DBHOST_JZ,
                          port=DBPORT_JZ,
                          user=DBUSER_JZ,
                          password=DBPWD_JZ,
                          db=DBNAME_JZ,
                          charset=DBCHAR_JZ)

    df = pd.read_sql(sql, con)
    con.close()

    return df


# 文件路径
local_path = r'D:\code\Alpha_Factor_Module\JZMF'
DATA_PATH = local_path + "/data/"

FACTOR_PATH = local_path + "/data/factor/"
SECTOR_PATH = local_path + "/data/sector/"

DB_PATH = local_path + "/data/database/"
DB_INFO_PATH = local_path + "/data/database/infodata/"
DB_TIME_PATH = local_path + "/data/database/timedata/"
DB_QUOTE_PATH = local_path + "/data/database/quotedata/"
DB_INDU_PATH = local_path + "/data/database/industrydata/"
DB_FIN_PATH = local_path + "/data/database/financialdata/"
DB_IDX_PATH = local_path + "/data/database/indexdata/"
DB_FUR_PATH = local_path + "/data/database/futuredata/"

JQ_PATH = local_path + "/data/jqdata/"
JQ_MIN_PATH = local_path + "/data/jqdata/minute/"

MIN_FACTOR_CSV_PATH = local_path + "/data/min_factor/"

# 日期格式
date_format = "%Y-%m-%d"

# 中文字体
font = FontProperties(fname='C:/Windows/Fonts/msyh.ttf', size=12)
