# -*- coding: utf-8 -*-
import os

import pandas as pd

import JZMF.lib.config as config
from JZMF.lib.HistoryMD import HistoryMD
from JZMF.lib.log import logging

his_md = HistoryMD()


### ---代码转换---
def getSecuCode(list_in):
    """
    通过InnerCode获取SecuCode
    """

    list_out = his_md.ISMAP.reindex(list_in).SecuCode.tolist()

    return list_out


def getInnerCode(list_in):
    """
    通过SecuCode获取InnerCode
    """

    list_out = his_md.SIMAP.reindex(list_in).InnerCode.tolist()
    return list_out


### ---时间工具---
def isRptDate(date):
    """
    是否为报告期
    date: str
    return: bool
    """

    date = pd.to_datetime(date)
    month = date.month
    day = date.day

    if month in [6, 9]:
        if day == 30:
            return True

    if month in [3, 12]:
        if day == 31:
            return True

    return False


def nextRptDate(date, is_include):
    """
    后一个报告期
    date: str
    is_include: bool
    return: str
    """

    date = pd.to_datetime(date)
    nextDay = date + pd.offsets.Day(1) - pd.offsets.Day(is_include)
    while not isRptDate(nextDay):
        nextDay = nextDay + pd.offsets.Day(1)

    return nextDay.strftime(config.date_format)


def preRptDate(date, is_include):
    """
    前一个报告期
    date: str
    return: str
    """

    date = pd.to_datetime(date)
    preDay = date - pd.offsets.Day(1) + pd.offsets.Day(is_include)
    while not isRptDate(preDay):
        preDay = preDay - pd.offsets.Day(1)

    return preDay.strftime(config.date_format)


def shiftRptDate(rptDate, num):
    """
    移动num个报告期
    rptDate: str
    num: int，负数(前推)/正数(后推)
    return: str
    """

    assert isRptDate(rptDate), "rptDate not exist: %s" % rptDate

    if num > 0:
        while num > 0:
            rptDate = nextRptDate(rptDate, is_include=False)
            num = num - 1
    else:
        while num < 0:
            rptDate = preRptDate(rptDate, is_include=False)
            num = num + 1

    return rptDate


def isAnnualDate(date):
    """
    是否年报日期
    date: str
    return: bool
    """

    date = pd.to_datetime(date)
    month = date.month
    day = date.day

    if month == 12 and day == 31:
        return True
    else:
        return False


def preAnnualDate(date, is_include):
    """
    前一个年报期
    date: str
    return: str
    """

    date = pd.to_datetime(date)
    preDay = date - pd.offsets.Day(1) + pd.offsets.Day(is_include)
    while not isAnnualDate(preDay):
        preDay = preDay - pd.offsets.Day(1)

    return preDay.strftime(config.date_format)


def isFirstQuarterDate(date):
    """
    是否一季报日期
    date: str
    return: bool
    """

    date = pd.to_datetime(date)
    month = date.month
    day = date.day

    if month == 3 and day == 31:
        return True
    else:
        return False


def isThirdQuarterDate(date):
    """
    是否三季报日期
    date: str
    return: bool
    """

    date = pd.to_datetime(date)
    month = date.month
    day = date.day

    if month == 9 and day == 30:
        return True
    else:
        return False


def isSemiRptDate(date):
    """
    是否半年报和年报日期
    date: str
    return: bool
    """

    date = pd.to_datetime(date)
    month = date.month

    if isRptDate(date) and month in [6, 12]:
        return True
    else:
        return False


def preSemiRptDate(date, is_include):
    """
    前一个半年报或年报日期，默认不包含
    date: str
    return: str
    """

    date = pd.to_datetime(date)
    preDay = date - pd.offsets.Day(1) + pd.offsets.Day(is_include)
    while not isSemiRptDate(preDay):
        preDay = preDay - pd.offsets.Day(1)

    return preDay.strftime(config.date_format)


def preMonthEnd(date, is_include):
    """
    前一个月底交易日，默认包含
    date: str
    return: str
    """

    date = pd.to_datetime(date)
    date = date - pd.offsets.Day(1) + pd.offsets.Day(is_include)

    s = pd.Series(his_md.MONTHEND)
    rt = s[s <= date.strftime(config.date_format)]

    return rt.iat[-1]


def near_tradeday(date, dtype):
    """
    取最近一个交易日
    date: str
    dtype：0 取前一个数(包含)，1 取前一个数(不包含)，2 取后一个数(包含)，3 取后一个数(不包含)
    return: str
    """

    s = pd.Series(his_md.TRADEDAYS)

    if dtype == 0:
        rt = s[s <= date]
        return rt.iat[-1]

    elif dtype == 1:
        rt = s[s < date]
        return rt.iat[-1]

    elif dtype == 2:
        rt = s[s >= date]
        return rt.iat[0]

    elif dtype == 3:
        rt = s[s > date]
        return rt.iat[0]


def make_step_range(begDate, endDate, step=20):
    """
    取固定步长的时间序列
    begDate: str 取后一个数(包含)
    endDate: str 取前一个数(包含)
    step：int 步长间隔
    return: list
    """

    assert isinstance(step, int) and step > 0, "make_step_range wrong step: %s" % step

    # 传入非交易日情况
    begt = near_tradeday(begDate, dtype=2)
    endt = near_tradeday(endDate, dtype=0)

    # 取全部交易日
    trade_days = pd.Series(his_md.TRADEDAYS)
    trade_days = trade_days[(trade_days >= begt) & (trade_days <= endt)].tolist()

    trade_days = [trade_days[i] for i in range(len(trade_days)) if i % step == 0]

    # 加上首尾
    if endt not in trade_days:
        trade_days.append(endt)

    return trade_days


def make_period_range(begDate, endDate, period="month", dtype="head"):
    """
    取固定周期的时间序列
    begDate: str 取后一个数(包含)
    endDate: str 取前一个数(包含)
    period：周期类型 ('day', 'week', 'month', 'quarter')
    dtype：head（期初）or tail（期末），大于日级别的时候生效
    return: list
    """

    # 传入非交易日情况
    begt = near_tradeday(begDate, dtype=2)
    endt = near_tradeday(endDate, dtype=0)

    # 取全部交易日
    trade_days = pd.Series(his_md.TRADEDAYS)
    trade_days = trade_days[(trade_days >= begt) & (trade_days <= endt)].tolist()

    if period == "day":
        return trade_days

    df = pd.DataFrame(index=pd.date_range(start=begt, end=endt))
    if period in ["month", "quarter"]:
        df['year'] = df.index.year
        df[period] = getattr(df.index, period)
        df.index = df.index.strftime(config.date_format)
        df = df.loc[trade_days]
        grouped = df.groupby(["year", period])
    elif period == "week":
        df['dayofweek'] = df.index.dayofweek
        df['num'] = range(df.shape[0])
        df['weeklabel'] = df['num'] - df['dayofweek']
        df.index = df.index.strftime(config.date_format)
        df = df.loc[trade_days]
        grouped = df.groupby("weeklabel")
    else:
        raise ValueError("make_period_range wrong period: %s" % period)

    # 期初 or 期末
    if dtype in ["head", "tail"]:
        df = getattr(grouped, dtype)(1)
    else:
        raise ValueError("make_period_range wrong dtype: %s" % dtype)

    trade_days = df.index.tolist()

    # 加上首尾
    if begt not in trade_days:
        trade_days.insert(0, begt)
    if endt not in trade_days:
        trade_days.append(endt)

    return trade_days


def count_tradeday(begDate, endDate):
    """
    获取begDate和endDate之间的交易日数目(包含边界)
    begDate: str
    begDate: str
    return: int
    """

    # 传入非交易日情况
    begt = near_tradeday(begDate, dtype=2)
    endt = near_tradeday(endDate, dtype=0)

    s = pd.Series(his_md.TRADEDAYS)
    s = s[(s >= begt) & (s <= endt)]

    return s.shape[0]


def shift_tradeday(tradeDate, num):
    """
    给定tradeDate，获取该日期前/后 num 个交易日对应的交易日
    tradeDate: str，必须是交易日 (不是交易日的通过 near_tradeday 转换)
    num: int 负数(前推)/正数(后推)
    return: str
    """

    assert tradeDate in his_md.TRADEDAYS, "tradeDate not exist: %s" % tradeDate
    s = pd.Series(his_md.TRADEDAYS)

    if num < 0:
        rt = s[s < tradeDate]
    else:
        rt = s[s >= tradeDate]

    return rt.iat[num]


def near_settleday(date, dtype):
    """
    取最近一个交割日
    date: str
    dtype：0 取前一个数(包含)，1 取前一个数(不包含)，2 取后一个数(包含)，3 取后一个数(不包含)
    return: str
    """

    s = pd.Series(his_md.SETTLEDAYSALL)

    if dtype == 0:
        rt = s[s <= date]
        return rt.iat[-1]

    elif dtype == 1:
        rt = s[s < date]
        return rt.iat[-1]

    elif dtype == 2:
        rt = s[s >= date]
        return rt.iat[0]

    elif dtype == 3:
        rt = s[s > date]
        return rt.iat[0]


### ---内部取值函数---
def _getSectorA(date):
    """
    取全部A股股票，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, "A", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_A.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 上市状态更改
        df = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "LC_ListStatus"))
        df = df[df.ChangeDate <= date]
        df = df.groupby("InnerCode").tail(1)
        df = df[df.ChangeType.isin([1, 2, 3, 6])]
        df = df.set_index("InnerCode")

        # 证券简称更名
        df1 = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "LC_SecuChange"))
        df1 = df1[df1.ChangeDate <= date]
        df1 = df1.groupby("InnerCode").tail(1)
        df1 = df1.set_index("InnerCode")

        # 加入名称
        df["SecuAbbr"] = df1["SecuAbbr"]

        # 存文件
        df = df[["SecuCode", "SecuAbbr"]]
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getSectorST(date):
    """
    取ST股票，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, "ST", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_ST.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 取全部A股
        df = _getSectorA(date)
        df["is_ST"] = df["SecuAbbr"].apply(lambda x: 1 if ("*" in x or "ST" in x or "退" in x) else 0)
        df = df[df.is_ST == 1]

        # 存文件
        df[["SecuCode", "SecuAbbr"]].to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getSectorSTIB(date):
    """
    取科创版股票，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, "STIB", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_STIB.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 取全部A股
        df = _getSectorA(date)
        df["is_STIB"] = df["SecuCode"].apply(lambda x: 1 if x.startswith('688') else 0)
        df = df[df.is_STIB == 1]

        # 存文件
        df[["SecuCode", "SecuAbbr"]].to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getIndexWeight(tradeDate, index_code):
    """
    取数据库的指数成分股权重
    返回pd.DataFrame
    """

    sql = "SELECT InnerCode, Weight AS weight FROM index_weight WHERE IndexCode = '%s' \
           AND Date = '%s' AND Flag = 3" % (index_code, tradeDate)

    df = config.query_jz(sql)

    # 整理
    df = df.set_index("InnerCode")

    return df


def _getSectorIndex(date, index_code):
    """
    取指数成分股，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, index_code, date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_%s.csv" % (date, index_code))
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 全部股票
        df_A = _getSectorA(date)

        # 取最近交易日(包含)
        tradeDate = near_tradeday(date, dtype=0)

        # 取指数权重
        df = _getIndexWeight(tradeDate, index_code)

        # 筛选
        df = df_A.reindex(df.index).dropna()

        # 存文件
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getIndustrySWOrg(date):
    """
    取申万行业，根据时间区分2011 or 2014标准，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, "industry_sw_org", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_industry.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 取全部股票
        df_A = _getSectorA(date)

        # 查询申万行业
        if date < "2014-01-01":
            df = pd.read_pickle(os.path.join(config.DB_INDU_PATH, "LC_ExgIndustry_SW_Old"))
        else:
            df = pd.read_pickle(os.path.join(config.DB_INDU_PATH, "LC_ExgIndustry_SW_New"))

        df = df[df['pubDate'] <= date]
        df = df.groupby("InnerCode").tail(1)
        df = df.set_index("InnerCode").reindex(df_A.index)
        df = df[["Lv1", "Lv2", "Lv3"]]

        # 合并
        df = pd.concat([df_A, df], axis=1, sort=False)

        # 存文件
        df.index.name = "InnerCode"
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getIndustrySWLv1(date):
    """
    取申万行业，统一取2014标准，仅支持一级行业分类，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, "industry_sw_lv1", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_industry.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 取全部股票
        df_A = _getSectorA(date)

        # 查询申万行业
        if date < "2014-01-01":
            df = pd.read_pickle(os.path.join(config.DB_INDU_PATH, "LC_ExgIndustry_SW_Old_Lv1"))
        else:
            df = pd.read_pickle(os.path.join(config.DB_INDU_PATH, "LC_ExgIndustry_SW_New"))

        df = df[df['pubDate'] <= date]
        df = df.groupby("InnerCode").tail(1)
        df = df.set_index("InnerCode").reindex(df_A.index)
        df = df[["Lv1"]]

        # 合并
        df = pd.concat([df_A, df], axis=1, sort=False)

        # 存文件
        df.index.name = "InnerCode"
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getIndustryCSRC(date):
    """
    取证监会行业，本地持久化
    返回pd.DataFrame
    """

    fileDir = os.path.join(config.SECTOR_PATH, "industry_csrc", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_industry.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        # 取全部股票
        df_A = _getSectorA(date)

        # 查询中信行业
        df = pd.read_pickle(os.path.join(config.DB_INDU_PATH, "LC_ExgIndustry_CSRC"))

        df = df[df['pubDate'] <= date]
        df = df.groupby("InnerCode").tail(1)
        df = df.set_index("InnerCode").reindex(df_A.index)
        df = df[["Lv1", "Lv2"]]

        # 合并
        df = pd.concat([df_A, df], axis=1, sort=False)

        # 存文件
        df.index.name = "InnerCode"
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def _getFinance(table, date, rptDate, fields):
    """
    取横截面财务数据
    返回pd.DataFrame
    """

    # 取全部数据
    df = his_md.sfinance(table, rptDate, fields)

    # 去除未发布数据
    df = df[df.pubDate <= date]

    # 如果有数据
    if not df.empty:
        # 取最新的一份
        df = df.groupby("InnerCode").apply(lambda x: x.fillna(method="pad").tail(1))

    # 重设索引
    df = df.set_index("InnerCode")

    return df


### ---筛选函数---
def filter_ST(date, codes):
    """
    剔除ST股票
    """

    STcodes = _getSectorST(date).index.tolist()
    reserved = set(codes) - set(STcodes)

    return reserved


def filter_STIB(date, codes):
    """
    剔除科创版股票，返回要留下的股票
    """

    STIBcodes = _getSectorSTIB(date).index.tolist()
    reserved = set(codes) - set(STIBcodes)

    return reserved


def filter_halt(date, codes):
    """
    剔除停牌股票
    """

    tradeDate = near_tradeday(date, dtype=0)

    amount = his_md.samount(tradeDate, tradeDate, codes).loc[tradeDate]

    reserved = amount[amount > 0].index

    return reserved


def filter_ipo1Y(date, codes):
    """
    剔除上市不到一年的股票
    """

    referDate = pd.to_datetime(date) - pd.offsets.Day(365)
    s = his_md.LISTDATE.ListedDate.loc[codes]

    reserved = s[s <= referDate.strftime(config.date_format)].index

    return reserved


def filter_highEqLow(date, codes):
    """
    剔除一字版的股票
    """

    tradeDate = near_tradeday(date, dtype=0)

    high = his_md.shigh(tradeDate, tradeDate, codes).loc[tradeDate]
    low = his_md.slow(tradeDate, tradeDate, codes).loc[tradeDate]
    spread = high - low

    reserved = spread[spread != 0].index

    return reserved


### ---对外取值函数---
def get_stock_return(begDate, endDate, codes):
    """
    计算股票的区间收益，一般行情软件通用方法
    返回pd.Series
    """

    # 收益率序列
    df = his_md.schange(begDate, endDate, codes)

    # 计算期间收益率
    values = df.drop(begDate).values
    values = (values + 1).prod(axis=0) - 1
    stock_return = pd.Series(values, index=codes)

    return stock_return


def get_price_change(begDate, endDate, codes, begPriceType='close', endPriceType='close'):
    """
    计算股票的区间收益，自定义起止价格
    返回pd.Series
    """

    # 价格序列
    begPrice = getattr(his_md, 's%s_adj' % begPriceType)(begDate, begDate, codes, baseDate=endDate).loc[begDate]
    endPrice = getattr(his_md, 's%s_adj' % endPriceType)(endDate, endDate, codes, baseDate=endDate).loc[endDate]

    # 计算期间收益率
    stock_return = endPrice / begPrice - 1

    return stock_return


def get_stock_industry(date, codes):
    """
    取申万一级行业分类
    返回pd.Series
    """

    df = _getIndustrySWLv1(date)

    return df["Lv1"].reindex(codes)


def get_index_weight(date, index_code, drop_halt=False):
    """
    取指数成分股权重
    返回pd.DataFrame
    """

    tradeDate = near_tradeday(date, dtype=0)
    df = _getIndexWeight(tradeDate, index_code)

    if drop_halt:
        df = df.loc[filter_halt(tradeDate, df.index)]

    logging.info("index weight %s: tradeDate=%s, len=%s, sum=%.3f", date, tradeDate, df.shape[0], df.weight.sum())

    return df


def getStockPool(date, index_code=None, filters=[]):
    """
    取板块股票池
    返回pd.DataFrame
    """

    if index_code is None:
        df = _getSectorA(date)
    else:
        df = _getSectorIndex(date, index_code)

    # 标记
    for fi in filters:
        df.loc[fi(date, df.index), fi.__name__] = 1

    return df


def getStockList(date, index_code=None, filters=[filter_ST, filter_halt, filter_highEqLow, filter_ipo1Y]):
    """
    取板块股票池
    返回list
    """

    df = getStockPool(date, index_code, filters)

    # 筛选
    codes = df.dropna().index.tolist()

    return codes


def getStockName(date, codes):
    """
    获取股票在指定交易日的中文名称
    返回pd.Series
    """

    df = _getSectorA(date)

    return df.SecuAbbr.reindex(codes)


def getIndustrySW(date, codes):
    """
    取申万行业分类
    返回pd.DataFrame
    """

    df = _getIndustrySWOrg(date)

    return df.reindex(codes)


def getIndustryCSRC(date, codes):
    """
    取证监会行业分类
    返回pd.DataFrame
    """

    df = _getIndustryCSRC(date)

    return df.reindex(codes)


def getShares(date, codes, fields):
    """
    取股票股本数量
    返回pd.DataFrame
    """

    # 查询股本变动记录
    df = his_md.STOCKSHARE

    # 整理
    df = df[(df['pubDate'] <= date) & (df['EndDate'] <= date)]
    df = df.groupby("InnerCode").tail(1)
    df = df.set_index("InnerCode")
    df = df[fields]

    # 对齐索引
    df = df.reindex(codes)
    df.insert(0, "SecuCode", getSecuCode(codes))
    df.insert(1, "date", date)

    return df


def getBalance(date, rptDate, codes, fields):
    """
    取资产负债表因子值
    返回pd.DataFrame
    """

    df = _getFinance("LC_BalanceSheetAll", date, rptDate, fields)

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)

    # 插入时间
    df.rptDate = rptDate
    df.insert(1, "date", date)

    return df


def getIncome(date, rptDate, codes, fields):
    """
    取利润表因子值
    返回pd.DataFrame
    """

    df = _getFinance("LC_IncomeStatementAll", date, rptDate, fields)

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)

    # 插入时间
    df.rptDate = rptDate
    df.insert(1, "date", date)

    return df


def getCash(date, rptDate, codes, fields):
    """
    取现金流量表因子值
    返回pd.DataFrame
    """

    df = _getFinance("LC_CashFlowStatementAll", date, rptDate, fields)

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)

    # 插入时间
    df.rptDate = rptDate
    df.insert(1, "date", date)

    return df


def getDerived(date, rptDate, codes, fields):
    """
    取衍生表因子值
    返回pd.DataFrame
    """

    df = _getFinance("LC_FSDerivedData", date, rptDate, fields)

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)

    # 插入时间
    df.rptDate = rptDate
    df.insert(1, "date", date)

    return df


def getForecast(date, rptDate, codes, fields):
    """
    取业绩预告因子值
    返回pd.DataFrame
    """

    df = _getFinance("LC_PerformanceForecast", date, rptDate, fields)

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)

    # 插入时间
    df.rptDate = rptDate
    df.insert(1, "date", date)

    return df


def getLetters(date, rptDate, codes, fields):
    """
    取业绩快报因子值
    返回pd.DataFrame
    """

    df = _getFinance("LC_FSPerformedLetters", date, rptDate, fields)

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)

    # 插入时间
    df.rptDate = rptDate
    df.insert(1, "date", date)

    return df


def getFactorTTM(date, rptDate, codes, factor, item):
    """
    取报告期因子的TTM值
    返回pd.DataFrame
    """

    df = factor.getReport(date, rptDate, codes)

    # 计算连续四季度累计值
    if not isAnnualDate(rptDate):
        preAnnDate = preAnnualDate(rptDate, is_include=False)
        df[item + 'LY'] = factor.getReport(date, preAnnDate, codes).value

        preYearDate = shiftRptDate(rptDate, -4)
        df[item + 'LYCP'] = factor.getReport(date, preYearDate, codes).value

        df[item + 'TTM'] = df[item] + df[item + 'LY'] - df[item + 'LYCP']
    else:
        df[item + 'TTM'] = df[item]

    return df


def getFactorQFA(date, rptDate, codes, factor, item):
    """
    取报告期因子的QFA值
    返回pd.DataFrame
    """

    df = factor.getReport(date, rptDate, codes)

    # 计算单季度值
    if not isFirstQuarterDate(rptDate):
        preRptDate_ = shiftRptDate(rptDate, -1)
        df[item + 'LP'] = factor.getReport(date, preRptDate_, codes).value

        df[item + 'QFA'] = df[item] - df[item + 'LP']
    else:
        df[item + 'QFA'] = df[item]

    return df


def getFactorYOY(date, rptDate, codes, factor, item):
    """
    取报告期因子的YOY值
    返回pd.DataFrame
    """

    df = factor.getReport(date, rptDate, codes)
    preYearDate = shiftRptDate(rptDate, -4)
    df[item + 'LYCP'] = factor.getReport(date, preYearDate, codes).value

    # 计算同比增长率
    df[item + 'YOY'] = (df[item] - df[item + 'LYCP']) / df[item + 'LYCP'].abs()

    return df


def getHKHoldings(date, codes):
    """
    取沪(深)港通持股统计，本地持久化
    返回pd.DataFrame
    """

    assert date >= "2017-03-17", "只能查询2017年3月17日或者之后的持股纪录"

    fileDir = os.path.join(config.SECTOR_PATH, "HKHoldings", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_HKHoldings.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        sql = "SELECT InnerCode, SecuCode, Percent, ShareNum FROM hkland_shares WHERE Date='%s'" % date
        df = config.query_jz(sql)
        if df.empty:
            raise Exception("empty DataBase: %s" % sql)

        df = df.set_index("InnerCode")

        # 存文件
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    # 对齐索引
    df = df.reindex(codes)
    df["SecuCode"] = getSecuCode(codes)
    df.insert(1, "date", date)

    return df


def _getMarginTarget(date):
    """
    取融资融券标的，本地持久化
    返回pd.DataFrame
    """

    assert date < '2020-04-20', '聚源数据权限已到期'

    fileDir = os.path.join(config.SECTOR_PATH, "MarginTarget", date.split("-")[0], date.split("-")[1])
    if not os.path.exists(fileDir):
        os.makedirs(fileDir)

    fileName = os.path.join(fileDir, "%s_MarginTarget.csv" % date)
    if not os.path.exists(fileName):
        logging.info('新建cache文件%s', fileName)

        sql = "SELECT InnerCode, DATE_FORMAT(InDate, '%Y-%m-%d') InDate \
        FROM MT_TargetSecurities WHERE TargetCategory = 20 \
        AND InDate <= '{0}' AND (OutDate > '{0}' OR OutDate IS NULL) \
        ORDER BY InnerCode, Indate".format(date)

        df = config.query_jy(sql)

        df = df.set_index("InnerCode")

        # 存文件
        df.to_csv(fileName, encoding='gbk')
    else:
        logging.info('基础cache文件%s', fileName)
        df = pd.read_csv(fileName, index_col="InnerCode", encoding='gbk')

    return df


def getMarginTarget(date):
    """
    取融资融券股票池
    返回list
    """

    df = _getMarginTarget(date)

    # 筛选
    codes = list(set(df.index))

    return codes