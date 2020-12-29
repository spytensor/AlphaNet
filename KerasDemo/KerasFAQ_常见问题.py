# 如何获取中间层的输出？

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as K
# import tensorflow as tf

import numpy as np

#%%
# 如何获取中间层的输出？

def method1():
    """
    Dense(units=64, input_dim=100, name='D1')
    input_dim   输入的维数
    :return:
    """
    model = Sequential()
    model.add(Dense(units=64, input_dim=100, name='D1'))
    model.add(Activation("relu"))
    model.add(Dense(units=10))
    model.add(Activation("softmax"))

    layer_name = 'D1'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model.predict(np.random.random((1, 100)))
    print(intermediate_output.shape)


def method2():
    """
    :return:
    """
    model = Sequential()
    model.add(Dense(units=64, input_dim=100, name='D1'))
    model.add(Activation("relu"))
    model.add(Dense(units=10))
    model.add(Activation("softmax"))

    # with a Sequential model
    get_3rd_layer_output = K.function([model.layers[0].input],
                                      [model.layers[3].output])
    X = np.random.random((1, 100))
    layer_output = get_3rd_layer_output([X])[0]
    print(layer_output.shape)

#%%
# 如何保存Keras模型？
def save_model():
    save_path = r''
    model = Sequential()
    model.add(Dense(units=64, input_dim=100, name='D1'))
    model.save(save_path)


if __name__=='__main__':
    method2()