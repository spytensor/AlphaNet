from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation


def tanh(x):
    return K.tanh(x)


model = Model()

model.add(Dense(64, activation=tanh))
model.add(Activation(tanh))
