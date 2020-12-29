from tensorflow.keras.layers import Layer
import numpy as np

class ts_mean(Layer):
    def __init__(self, **kwargs):
        self.window = 10
        self.stride = 3
        self.features_num = 8
        self.last_layer_stride = 10
        self.backward_len = 30/self.last_layer_stride
        self.logging = True
        self.input_data_shape = (None, self.features_num, self.backward_len, 1)
        super(ts_mean, self).__init__(**kwargs)


    def call(self, inputs, **kwargs):
        assert inputs.shape[1:] == self.input_data_shape[1:]
        arr_mean = np.mean(inputs,axis=2)
        return np.reshape(arr_mean,newshape=(arr_mean.shape[0],arr_mean.shape[1],int(self.backward_len/self.stride),1))


    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]


class ts_max(Layer):
    def __init__(self, **kwargs):
        self.window = 10
        self.stride = 3
        self.features_num = 8
        self.last_layer_stride = 10
        self.backward_len = 30 / self.last_layer_stride
        self.logging = True
        self.input_data_shape = (None, self.features_num, self.backward_len, 1)
        super(ts_max, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert inputs.shape[1:] == self.input_data_shape[1:]
        arr_max = np.max(inputs, axis=2)
        return np.reshape(arr_max,
                          newshape=(arr_max.shape[0], arr_max.shape[1], int(self.backward_len / self.stride), 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]


class ts_min(Layer):
    def __init__(self, **kwargs):
        self.window = 10
        self.stride = 3
        self.features_num = 8
        self.last_layer_stride = 10
        self.backward_len = 30 / self.last_layer_stride
        self.logging = True
        self.input_data_shape = (None, self.features_num, self.backward_len, 1)
        super(ts_min, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        assert inputs.shape[1:] == self.input_data_shape[1:]
        arr_min = np.min(inputs, axis=2)
        return np.reshape(arr_min,
                          newshape=(arr_min.shape[0], arr_min.shape[1], int(self.backward_len / self.stride), 1))

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]