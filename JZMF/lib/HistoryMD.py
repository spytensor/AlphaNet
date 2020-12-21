# -*- coding: utf-8 -*-
from collections import defaultdict

import pandas as pd

import JZMF.lib.config as config
from JZMF.lib.log import *


class HistoryMD(object):
    """历史数据内存取值"""

    def __init__(self):
        try:
            # 日期常量
            self.TRADEDAYS = pd.read_pickle(os.path.join(config.DB_TIME_PATH, "TradeDays"))
            self.MONTHEND = pd.read_pickle(os.path.join(config.DB_TIME_PATH, "MonthEnd"))
            self.LASTDATE = self.TRADEDAYS[-1]
            self.SETTLEDAYSALL = pd.read_pickle(os.path.join(config.DB_TIME_PATH, "SettleDaysAll"))

            # 映射表
            self.SIMAP = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "StockMain"))[
                ["SecuCode", "InnerCode"]].set_index("SecuCode")
            self.ISMAP = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "StockMain"))[
                ["SecuCode", "InnerCode"]].set_index("InnerCode")

            # 股票基础
            self.LISTDATE = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "StockMain"))[
                ["InnerCode", "ListedDate"]].set_index("InnerCode")
            self.STOCKSHARE = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "LC_ShareStru"))
            self.STOCKDIVIDEND = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "LC_Dividend"))
            self.LENDINGSECU = pd.read_pickle(os.path.join(config.DB_INFO_PATH, "SMTCN_EXCH_SECU")).InnerCode.tolist()

            # 股票行情
            self.closedata = pd.DataFrame()
            self.opendata = pd.DataFrame()
            self.highdata = pd.DataFrame()
            self.lowdata = pd.DataFrame()
            self.changedata = pd.DataFrame()
            self.changedata_org = pd.DataFrame()
            self.amountdata = pd.DataFrame()
            self.amountdata_org = pd.DataFrame()
            self.volumedata = pd.DataFrame()
            self.volumedata_org = pd.DataFrame()
            self.turndata = pd.DataFrame()
            self.turndata_org = pd.DataFrame()
            self.totalsharesdata = pd.DataFrame()
            self.adjustfactordata = pd.DataFrame()

            # 指数行情
            self.index_opendata = pd.DataFrame()
            self.index_closedata = pd.DataFrame()
            self.index_changedata = pd.DataFrame()

            # 公司财务
            self.financedata = defaultdict(pd.DataFrame)
        except Exception as e:
            logging.info(e)

    def sclose(self, begDate, endDate, codes):
        """
        获取股票的历史收盘价，未复权
        """

        if self.closedata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "close")
            self.closedata = pd.read_pickle(filepath)

        return self.closedata.loc[begDate:endDate, codes]

    def sopen(self, begDate, endDate, codes):
        """
        获取股票的历史开盘价，未复权
        """

        if self.opendata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "open")
            self.opendata = pd.read_pickle(filepath)

        return self.opendata.loc[begDate:endDate, codes]

    def shigh(self, begDate, endDate, codes):
        """
        获取股票的历史最高价，未复权
        """

        if self.highdata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "high")
            self.highdata = pd.read_pickle(filepath)

        return self.highdata.loc[begDate:endDate, codes]

    def slow(self, begDate, endDate, codes):
        """
        获取股票的历史最低价，未复权
        """

        if self.lowdata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "low")
            self.lowdata = pd.read_pickle(filepath)

        return self.lowdata.loc[begDate:endDate, codes]

    def schange(self, begDate, endDate, codes):
        """
        获取股票的历史涨跌幅，填充nan为0
        """

        if self.changedata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "change")
            self.changedata = pd.read_pickle(filepath).fillna(0) / 100

        return self.changedata.loc[begDate:endDate, codes]

    def schange_org(self, begDate, endDate, codes):
        """
        获取股票的历史涨跌幅，不填充nan
        """

        if self.changedata_org.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "change")
            self.changedata_org = pd.read_pickle(filepath) / 100

        return self.changedata_org.loc[begDate:endDate, codes]

    def samount(self, begDate, endDate, codes):
        """
        获取股票的历史成交金额，填充nan为0
        """

        if self.amountdata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "amount")
            self.amountdata = pd.read_pickle(filepath).fillna(0)

        return self.amountdata.loc[begDate:endDate, codes]

    def samount_org(self, begDate, endDate, codes):
        """
        获取股票的历史成交金额，不填充nan
        """

        if self.amountdata_org.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "amount")
            self.amountdata_org = pd.read_pickle(filepath)

        return self.amountdata_org.loc[begDate:endDate, codes]

    def svolume(self, begDate, endDate, codes):
        """
        获取股票的历史成交量，填充nan为0
        """

        if self.volumedata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "volume")
            self.volumedata = pd.read_pickle(filepath).fillna(0)

        return self.volumedata.loc[begDate:endDate, codes]

    def svolume_org(self, begDate, endDate, codes):
        """
        获取股票的历史成交量，不填充nan
        """

        if self.volumedata_org.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "volume")
            self.volumedata_org = pd.read_pickle(filepath)

        return self.volumedata_org.loc[begDate:endDate, codes]

    def sturn(self, begDate, endDate, codes):
        """
        获取股票的历史换手率，填充nan为0
        """

        if self.turndata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "turn")
            self.turndata = pd.read_pickle(filepath).fillna(0)

        return self.turndata.loc[begDate:endDate, codes]

    def sturn_org(self, begDate, endDate, codes):
        """
        获取股票的历史换手率，不填充nan
        """

        if self.turndata_org.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "turn")
            self.turndata_org = pd.read_pickle(filepath)

        return self.turndata_org.loc[begDate:endDate, codes]

    def stotalshares(self, begDate, endDate, codes):
        """
        获取股票的总股本
        """

        if self.totalsharesdata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "totalshares")
            self.totalsharesdata = pd.read_pickle(filepath)

        return self.totalsharesdata.loc[begDate:endDate, codes]

    def sadjustfactor(self, begDate, endDate, codes):
        """
        获取股票的复权因子
        """

        if self.adjustfactordata.empty:
            filepath = os.path.join(config.DB_QUOTE_PATH, "adjustfactor")
            self.adjustfactordata = pd.read_pickle(filepath)

        return self.adjustfactordata.loc[begDate:endDate, codes]

    def sclose_adj(self, begDate, endDate, codes, baseDate=None):
        """
        获取股票的历史收盘价，动态前复权，如果复权基准日没填，则默认为结束日期
        """

        if baseDate is None:
            baseDate = endDate

        # 未复权价格
        closeDf = self.sclose(begDate, endDate, codes)

        # 复权因子
        refAdjFac = self.sadjustfactor(baseDate, baseDate, codes).loc[baseDate]
        adjfacDf = self.sadjustfactor(begDate, endDate, codes)

        return (closeDf * adjfacDf) / refAdjFac

    def sopen_adj(self, begDate, endDate, codes, baseDate=None):
        """
        获取股票的历史开盘价，动态前复权，如果复权基准日没填，则默认为结束日期
        """

        if baseDate is None:
            baseDate = endDate

        # 未复权价格
        openDf = self.sopen(begDate, endDate, codes)

        # 复权因子
        refAdjFac = self.sadjustfactor(baseDate, baseDate, codes).loc[baseDate]
        adjfacDf = self.sadjustfactor(begDate, endDate, codes)

        return (openDf * adjfacDf) / refAdjFac

    def shigh_adj(self, begDate, endDate, codes, baseDate=None):
        """
        获取股票的历史最高价，动态前复权，如果复权基准日没填，则默认为结束日期
        """

        if baseDate is None:
            baseDate = endDate

        # 未复权价格
        highDf = self.shigh(begDate, endDate, codes)

        # 复权因子
        refAdjFac = self.sadjustfactor(baseDate, baseDate, codes).loc[baseDate]
        adjfacDf = self.sadjustfactor(begDate, endDate, codes)

        return (highDf * adjfacDf) / refAdjFac

    def slow_adj(self, begDate, endDate, codes, baseDate=None):
        """
        获取股票的历史收盘价，动态前复权，如果复权基准日没填，则默认为结束日期
        """

        if baseDate is None:
            baseDate = endDate

        # 未复权价格
        lowDf = self.slow(begDate, endDate, codes)

        # 复权因子
        refAdjFac = self.sadjustfactor(baseDate, baseDate, codes).loc[baseDate]
        adjfacDf = self.sadjustfactor(begDate, endDate, codes)

        return (lowDf * adjfacDf) / refAdjFac

    def sindexopen(self, begDate, endDate, codes):
        """
        获取指数的历史收盘价
        """

        if self.index_opendata.empty:
            fileName = os.path.join(config.DB_IDX_PATH, "index_open")
            self.index_opendata = pd.read_pickle(fileName)

        return self.index_opendata.loc[begDate:endDate, codes]

    def sindexclose(self, begDate, endDate, codes):
        """
        获取指数的历史收盘价
        """

        if self.index_closedata.empty:
            fileName = os.path.join(config.DB_IDX_PATH, "index_close")
            self.index_closedata = pd.read_pickle(fileName)

        return self.index_closedata.loc[begDate:endDate, codes]

    def sindexchange(self, begDate, endDate, codes):
        """
        获取指数的历史涨跌幅
        """

        if self.index_changedata.empty:
            filepath = os.path.join(config.DB_IDX_PATH, "index_close")
            self.index_changedata = pd.read_pickle(filepath).pct_change()

        return self.index_changedata.loc[begDate:endDate, codes]

    def sfinance(self, table, rptDate=None, fields=None, codes=None):
        """
        获取历史财务数据
        table: 聚源数据字典表
        rptdDate: str
        fields: list
        codes: list
        返回：pd.DataFrame
        """

        if self.financedata[table].empty:
            fileName = os.path.join(config.DB_FIN_PATH, table)
            self.financedata[table] = pd.read_pickle(fileName)

        df = self.financedata[table]
        if rptDate is not None:
            df = df[df.rptDate == rptDate]
        if fields is not None:
            df = df[["InnerCode", "SecuCode", "rptDate", "pubDate"] + fields]
        if codes is not None:
            df = df[df["InnerCode"].isin(codes)]

        return df


his_md = HistoryMD()
