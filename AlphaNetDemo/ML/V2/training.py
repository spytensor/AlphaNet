# from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.layers import Input, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
# import numpy as np

from AlphaNetDemo.data_process import generate_one_day_xy
from JZMF.lib.utility import make_period_range
from AlphaNetDemo.ML.V2.MyConvolutionalLayers import (ts_zscore, ts_decaylinear, ts_return, ts_stddev)
from AlphaNetDemo.ML.V2.MyPoolingLayer import (ts_min, ts_max, ts_mean)
import tensorflow


tensorflow.compat.v1.enable_eager_execution()

model = Model()
main_input = Input(shape=(8, 30, 1), dtype='int32', name='main_input')


# 卷积层
x1 = ts_zscore()(main_input)
x2 = ts_decaylinear()(main_input)
x3 = ts_return()(main_input)
x4 = ts_stddev()(main_input)

xx = tensorflow.keras.layers.concatenate([x1, x2, x3, x4])

xx = BatchNormalization()(xx)

# 池化层
xx1 = ts_max()(xx)
xx2 = ts_min()(xx)
xx3 = ts_mean()(xx)

xxx = tensorflow.keras.layers.concatenate([xx1, xx2, xx3])

xxx = BatchNormalization()(xxx)

fxxx = Flatten()(xxx)
dxxx1 = Dense(30, activation='relu', kernel_initializer='truncated_normal')(fxxx)
drxxx = Dropout(0.5)(dxxx1)
dxxx2 = Dense(1, activation='linear')(drxxx)
model.summary()


def train_generator():
    """
    Generates batch and batch and batch then feed into models.
    :return:
    """
    begDate = '2017-01-01'
    endDate = '2020-01-01'
    period = 'week'
    dtype = 'tail'
    interval = 2

    trade_date = make_period_range(begDate, endDate, period, dtype)[::interval]
    while True:
        for date in trade_date:
            x, y, _ = generate_one_day_xy(date)
            yield x, y


def test_generator():
    """
    Generates batch and batch and batch then feed into models.
    :return:
    """
    begDate = '2020-01-01'
    endDate = '2020-01-15'
    period = 'week'
    dtype = 'tail'
    interval = 2

    trade_date = make_period_range(begDate, endDate, period, dtype)[::interval]
    while True:
        for date in trade_date:
            x, y, _ = generate_one_day_xy(date)
            yield x, y


ad = RMSprop(lr=0.001)
model.compile(optimizer=ad, loss='mse')

steps_per_epoch = 10
epochs = 5
week_per_year = 50
interval_ = 2
years = 3

history = model.fit_generator(generator=train_generator(),
                              steps_per_epoch=steps_per_epoch,
                              epochs=int(years * epochs * week_per_year / interval_ / steps_per_epoch),
                              # validation_data=test_generator(),
                              )

model.save('AlphaModel')

result = model.test_on_batch(generate_one_day_xy('2020-07-24')[0], generate_one_day_xy('2020-07-24')[1])
print(result)
