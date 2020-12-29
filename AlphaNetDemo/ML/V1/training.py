from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import RMSprop
from AlphaNetDemo.data_process import generate_one_day_xy
from JZMF.lib.utility import make_period_range


model = Sequential()
model.add(Input((8, 30, 1)))
model.add(Conv2D(filters=7, strides=10, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=3, padding='same'))
model.add(Flatten())
model.add(Dense(30, activation='relu', kernel_initializer='truncated_normal'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
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
