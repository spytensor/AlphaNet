import JZMF.lib.utility as utility
from JZMF.lib.HistoryMD import his_md

import pandas as pd
import os

PROJECT_PATH = r'D:\code\AlphaNet'

def generate_one_xy(date, backward_window=30, forward_window=10, if_logging=False):
    """
    https://www.jianshu.com/p/fc2fe026f002
    :param if_logging:
    :param date:
    :param backward_window:
    :param forward_window:
    :return:
    """
    if not os.path.exists(PROJECT_PATH+r'\db\{}'.format(date)):
        os.makedirs(PROJECT_PATH+r'\db\{}'.format(date))

    if os.path.exists(PROJECT_PATH+r'\db\{}\xdata_b{}_f{}.pkl'.format(date,backward_window,forward_window)):
        return pd.read_pickle(PROJECT_PATH+r'\db\{}\xdata_b{}_f{}.pkl'.format(date,backward_window,forward_window)), \
               pd.read_pickle(PROJECT_PATH + r'\db\{}\ydata_b{}_f{}.pkl'.format(date,backward_window, forward_window))

    begDate = utility.shift_tradeday(date, -backward_window + 1)
    endDate = date

    codes = utility.getStockList(date)
    open_df = his_md.sopen(begDate, endDate, codes)
    high_df = his_md.shigh(begDate, endDate, codes)
    low_df = his_md.slow(begDate, endDate, codes)
    close_df = his_md.sclose(begDate, endDate, codes)
    turn_df = his_md.sturn(begDate, endDate, codes)
    change_df = his_md.schange(begDate, endDate, codes)
    vol_df = his_md.svolume(begDate, endDate, codes)
    vwap_df = his_md.samount(begDate, endDate, codes) / vol_df

    df_list = [open_df, high_df, low_df, close_df, turn_df, change_df, vol_df, vwap_df]
    data_name_list = ['open', 'high', 'low', 'close', 'turn', 'change', 'vol', 'vwap']
    midf_list = list()
    for index, df in enumerate(df_list):
        midf = df.copy()
        data_name = data_name_list[index]
        midf.index = pd.MultiIndex.from_product([[data_name], midf.index])
        midf_list.append(midf)
    micdf = pd.concat(midf_list, axis=0, sort=True)
    pd.to_pickle(micdf,PROJECT_PATH+r'\db\{}\micdf_b{}_f{}.pkl'.format(date,backward_window,forward_window))



    micdf.stock_num = micdf.shape[1]
    micdf.days_num = backward_window
    micdf.field_num = len(data_name_list)



    x_data = micdf.T.values.reshape(micdf.stock_num, micdf.field_num, micdf.days_num,1, order='C')
    pd.to_pickle(x_data,PROJECT_PATH+r'\db\{}\xdata_b{}_f{}.pkl'.format(date,backward_window,forward_window))


    y_data = utility.get_price_change(date, utility.shift_tradeday(date, forward_window), codes=micdf.columns).values
    pd.to_pickle(y_data,PROJECT_PATH+r'\db\{}\ydata_b{}_f{}.pkl'.format(date,backward_window,forward_window))


    if if_logging:
        print(micdf)
        print(micdf.T.values)
        print(x_data)
        print(x_data[0])
        print(x_data.shape)
        print(y_data.shape)

    return x_data, y_data


if __name__ == '__main__':
    generate_one_xy('2020-10-30')
