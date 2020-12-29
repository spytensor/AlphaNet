import tensorflow
import pandas as pd

from AlphaNetDemo.data_process import generate_one_day_xy
from JZMF.lib.MyFactorClass import MyFactor
import JZMF.lib.utility as utility

model = tensorflow.keras.models.load_model('AlphaModel')

begDate = '2017-01-02'
endDate = '2020-10-01'


def predict_func(date, codes):
    x_data, _, micdf = generate_one_day_xy(date)
    micdf.dropna(axis=1, inplace=True)

    predict_y = model.predict(x_data)
    predict_y = pd.Series(predict_y.reshape(len(predict_y)),index=micdf.columns)

    df = pd.DataFrame()
    df['value'] = predict_y
    df.index.name = 'InnerCode'
    df = df.reindex(codes)
    return df


fac_alphanet = MyFactor('AlphaNet', predict_func, fclass='ML', saveFile=True)
fac_alphanet.setStoragePath('')

if __name__=='__main__':
    """
    
    """
    for dt in utility.make_period_range(begDate,endDate,'week','tail')[::2]:
        fac_alphanet.getValue(dt)

    pass