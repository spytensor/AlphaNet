from unittest import TestCase
import pandas as pd

from .MyConvolutionalLayers import ts_stddev,ts_decaylinear,ts_zscore,ts_return


class Testts_stddev(TestCase):
    def test_ts_stddev_demo(self):
        """
        有点出入，但是先不管了
        :return:
        """
        x_data = pd.read_pickle(r'../../db/2017-01-03/xdata_b30_f10.pkl')
        ts_s = ts_stddev()
        output = ts_s(x_data)
        print(output.shape)
        temp1 = output[0, :, :, 0]
        print(temp1)
        temp2 = pd.DataFrame(x_data[0, :, :, 0])
        temp2 = temp2.rolling(window=10, axis=1).std().T
        print(temp2)

    pass

    def test_call(self):
        x_data = pd.read_pickle(r'../../db/2017-01-03/xdata_b30_f10.pkl')
        ts_s = ts_stddev()
        y_data = ts_s(x_data)
        print(y_data.shape)


class Testts_zscore(TestCase):
    def test_call(self):
        x_data = pd.read_pickle(r'../../db/2017-01-03/xdata_b30_f10.pkl')
        ts_s = ts_zscore()
        output = ts_s(x_data)
        print(output.shape)


class Testts_decaylinear(TestCase):
    def test_call(self):
        x_data = pd.read_pickle(r'../../db/2017-01-03/xdata_b30_f10.pkl')
        ts_s = ts_decaylinear()
        output = ts_s(x_data)
        print(output.shape)


class Testts_return(TestCase):
    def test_call(self):
        x_data = pd.read_pickle(r'../../db/2017-01-03/xdata_b30_f10.pkl')
        ts_s = ts_return()
        output = ts_s(x_data)
        print(output.shape)

