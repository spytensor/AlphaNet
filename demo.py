from abc import ABC
# import tensorflow as tf
# import os
# import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from .data_process import generate_one_xy


def demo():
    """22222
    :return:
    """
    x_data, y_data = generate_one_xy('2020-10-30')
    #print(x_data.shape)
    # x_data[x_data<=0] = 0
    # y_data[y_data<=0] = 0

    (x_train, y_train), (x_test, y_test) = (x_data[:3000], y_data[:3000]), (x_data[3000:], y_data[3000:])


    class Baseline(Model, ABC):
        def __init__(self):
            super(Baseline, self).__init__()
            self.c1 = Conv2D(filters=6, kernel_size=(3, 3), padding='same')  # 卷积层
            self.b1 = BatchNormalization()  # BN层
            self.a1 = Activation('relu')  # 激活层
            self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
            self.d1 = Dropout(0.2)  # dropout层

            self.flatten = Flatten()
            self.f1 = Dense(128, activation='relu')
            self.d2 = Dropout(0.2)
            self.f2 = Dense(1)

        def call(self, x, training=None, mask=None):
            print(x)
            x = self.c1(x)
            x = self.b1(x)
            x = self.a1(x)
            x = self.p1(x)
            x = self.d1(x)

            x = self.flatten(x)
            x = self.f1(x)
            x = self.d2(x)
            y = self.f2(x)
            print(y)
            return y

    model = Baseline()
    ad = Adam(lr=0.00001,beta_1=0.9,beta_2=0.999,epsilon= 1e-8)
    model.compile(optimizer=ad,
                  loss='mse',
                  metrics= 'acc'
                  )

    history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test),
                        validation_freq=1, )
    model.summary()

    # 显示训练集和验证集的acc和loss曲线
    """
    这里是画图报错
    """
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def custom_layer(if_logging=True):
    """
    这里我检查了各个层的过程
    :param if_logging:
    :return:
    """
    x_data, y_data = generate_one_xy('2020-10-30')
    (x_train, y_train), (_, __) = (x_data[:3000], y_data[:3000]), (x_data[3000:], y_data[3000:])
    c1 = Conv2D(filters=6, kernel_size=(3, 3), padding='same')  # 卷积层
    b1 = BatchNormalization()  # BN层
    a1 = Activation('relu')  # 激活层
    p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  # 池化层
    d1 = Dropout(0.2)  # dropout层

    flatten = Flatten()
    f1 = Dense(128, activation='relu')
    d2 = Dropout(0.2)
    f2 = Dense(1)

    x1 = c1(x_train)
    x2 = b1(x1)
    x3 = a1(x2)
    x4 = p1(x3)
    x5 = d1(x4)

    if if_logging:
        print(x_train.shape)
        print(x4.shape)
        print(x5.shape)  # (3000, 4, 2, 6)

    x6 = flatten(x5)
    x7 = f1(x6)
    x8 = d2(x7)
    x9 = f2(x8)
    if if_logging:
        print(x6.shape)  # (3000, 48)
        print(x7.shape)
        print(x8.shape)
        print(x9.shape)

        print(x5)
        print(x6)
        print(x7)
        print(x8)
        print(x9)


if __name__ == '__main__':
    demo()
    # custom_layer(False)
