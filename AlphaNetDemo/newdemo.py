
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Input,Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model,Sequential
from tensorflow.keras.optimizers import Adam
from data_process import generate_one_xy



model = Sequential()
model.add(Input((8,30,1)))
model.add(Conv2D(filters=6, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='sigmoid'))#还是用sigmod激活，对应0~1归一化
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

x = pd.read_pickle("xdata_b30_f10.pkl")
y = pd.read_pickle("ydata_b30_f10.pkl")

#对所有x归一化
nx = np.zeros(np.shape(x))
for i in range(len(x)):
    sx = x[i,:,:,0]
    for j in range(len(sx)):
        ssx = sx[j,:]        
        snx = (ssx-np.min(ssx))/(np.max(ssx)-np.min(ssx))
        nx[i,j,:,0] = snx   
#y也归一化一遍 
ny = (y+1)/2

(x_train, y_train), (x_test, y_test) = (nx[:3000], ny[:3000]), (nx[3000:], ny[3000:])

ad = Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon= 1e-8)
model.compile(optimizer=ad,loss='mse')

history = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test),
                    validation_freq=1)

 
