from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

model = Model()
model.add(Dense(64, W_constraint = max_norm(2)))