from tensorflow.keras.layers import Layer
import numpy as np
import pandas as pd
import tensorflow

# class ts_stddev(Layer):
#     def __init__(self, **kwargs):
#         self.window = 10
#         self.stride = 10
#         self.features_num = 8
#         self.backward_len = 30
#         self.logging = False
#         self.input_data_shape = (None, self.features_num, self.backward_len, 1)
#         super(ts_stddev, self).__init__(**kwargs)
#
#     def call(self, inputs, **kwargs):
#         assert inputs.shape[1:] == self.input_data_shape[1:]
#         arr = inputs.numpy()
#         arr_r10 = np.roll(arr, shift=self.window, axis=2)
#
#         temp_dict = dict()
#         for num in range(int(self.backward_len / self.stride)):
#             arr_trim = arr_r10[:, :, num * self.stride:(num + 1) * self.stride, :]
#             arr_std10 = np.std(arr_trim, axis=2)
#             arr_std10_re = np.reshape(arr_std10, (arr_std10.shape[0], arr_std10.shape[1], 1, arr_std10.shape[2]))
#             if self.logging:
#                 print(num)
#                 print(arr_trim.shape)
#                 print(arr_std10.shape)
#                 print(arr_std10_re.shape)
#                 print(arr_trim[0, :, :, 0].shape)
#                 print(pd.DataFrame(arr_trim[0, :, :, 0]))
#                 print(np.std(arr_trim[0, :, :, 0], axis=1))
#             temp_dict[num] = arr_std10_re
#
#         total_num = int(self.backward_len / self.stride)
#         temp_list = [temp_dict[num] for num in range(1, total_num)]
#         temp_list.append(temp_dict[0])
#
#         result = np.concatenate(tuple(temp_list), axis=2)
#         return result
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]


class ts_zscore(Layer):
    def __init__(self, **kwargs):
        self.window = 10
        self.stride = 10
        self.features_num = 8
        self.backward_len = 30
        self.logging = False
        self.input_data_shape = (None, self.features_num, self.backward_len, 1)
        super(ts_zscore, self).__init__(**kwargs)

    def np_func(self,inputs):
        assert inputs.shape[1:] == self.input_data_shape[1:]
        arr = inputs.numpy()
        arr_r10 = np.roll(arr, shift=self.window, axis=2)

        temp_dict = dict()
        for num in range(int(self.backward_len / self.stride)):
            arr_trim = arr_r10[:, :, num * self.stride:(num + 1) * self.stride, :]
            arr_mean10 = np.mean(arr_trim, axis=2)
            arr_std10 = np.std(arr_trim, axis=2)
            arr_zscore10 = arr_mean10 / arr_std10
            arr_zscore10_re = np.reshape(arr_zscore10,
                                         (arr_zscore10.shape[0], arr_zscore10.shape[1], 1, arr_zscore10.shape[2]))
            if self.logging:
                print(num)
                print(arr_trim.shape)
                print(arr_std10.shape)
                print(arr_zscore10_re.shape)
                print(arr_trim[0, :, :, 0].shape)
                print(pd.DataFrame(arr_trim[0, :, :, 0]))
                print(np.std(arr_trim[0, :, :, 0], axis=1))
            temp_dict[num] = arr_zscore10_re

        total_num = int(self.backward_len / self.stride)
        temp_list = [temp_dict[num] for num in range(1, total_num)]
        temp_list.append(temp_dict[0])

        result = np.concatenate(tuple(temp_list), axis=2)
        return result

    def call(self, inputs, **kwargs):
        return self.np_func(inputs)


    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]


# class ts_return(Layer):
#     def __init__(self, **kwargs):
#         self.window = 10
#         self.stride = 10
#         self.features_num = 8
#         self.backward_len = 30
#         self.logging = False
#         self.input_data_shape = (None, self.features_num, self.backward_len, 1)
#         super(ts_return, self).__init__(**kwargs)
#
#     def call(self, inputs, **kwargs):
#         assert inputs.shape[1:] == self.input_data_shape[1:]
#
#         arr = inputs.numpy()
#         arr_r10 = np.roll(arr, shift=self.window, axis=2)
#
#         temp_dict = dict()
#         for num in range(int(self.backward_len / self.stride)):
#             arr_trim = arr_r10[:, :, num * self.stride:(num + 1) * self.stride, :]
#             arr_head10 = arr_trim[:, :, [0], :]
#             arr_tail10 = arr_trim[:, :, [-1], :]
#             arr_ret10 = arr_tail10 / arr_head10 - 1
#             # arr_zscore10_re = np.reshape(arr_ret10,
#             #                              (arr_ret10.shape[0], arr_ret10.shape[1], 1, arr_ret10.shape[2]))
#
#             arr_ret10_re = arr_ret10
#             if self.logging:
#                 print(num)
#                 print(arr_trim.shape)
#                 print(arr_ret10_re.shape)
#                 print(arr_trim[0, :, :, 0].shape)
#                 print(pd.DataFrame(arr_trim[0, :, :, 0]))
#                 print(np.std(arr_trim[0, :, :, 0], axis=1))
#             temp_dict[num] = arr_ret10_re
#
#         total_num = int(self.backward_len / self.stride)
#         temp_list = [temp_dict[num] for num in range(1, total_num)]
#         temp_list.append(temp_dict[0])
#
#         result = np.concatenate(tuple(temp_list), axis=2)
#         return result
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]


# class ts_decaylinear(Layer):
#     def __init__(self, **kwargs):
#         self.window = 10
#         self.stride = 10
#         self.features_num = 8
#         self.backward_len = 30
#         self.logging = False
#         self.input_data_shape = (None, self.features_num, self.backward_len, 1)
#         super(ts_decaylinear, self).__init__(**kwargs)
#
#     def call(self, inputs, **kwargs):
#         assert inputs.shape[1:] == self.input_data_shape[1:]
#
#         arr = inputs.numpy()
#         arr_r10 = np.roll(arr, shift=self.window, axis=2)
#         # 生成长度为30的权重向量
#         weight_arr = np.array(range(1, 1 + self.stride))
#         weight_arr = weight_arr / weight_arr.sum()
#         weight_arr2d = np.expand_dims(weight_arr, axis=0)
#         weight_arr2d = np.repeat(weight_arr2d, repeats=self.features_num, axis=0)
#         weight_arr3d = np.expand_dims(weight_arr2d, axis=0)
#         weight_arr3d = np.repeat(weight_arr3d, repeats=inputs.shape[0], axis=0)
#         weight_arr4d = np.reshape(weight_arr3d,
#                                   newshape=(inputs.shape[0], weight_arr3d.shape[1], weight_arr3d.shape[2], 1))
#
#         temp_dict = dict()
#         for num in range(int(self.backward_len / self.stride)):
#             arr_trim = arr_r10[:, :, num * self.stride:(num + 1) * self.stride, :]
#             assert arr_trim.shape == weight_arr4d.shape
#             arr_weight = arr_trim * weight_arr4d
#             arr_wsum = arr_weight.sum(axis=2)
#
#             arr_ret10_re = np.reshape(arr_wsum, newshape=(arr_wsum.shape[0], arr_wsum.shape[1], arr_wsum.shape[2], 1))
#             if self.logging:
#                 print(num)
#                 print(arr_trim.shape)
#                 print(arr_ret10_re.shape)
#                 print(arr_trim[0, :, :, 0].shape)
#                 print(pd.DataFrame(arr_trim[0, :, :, 0]))
#                 print(np.std(arr_trim[0, :, :, 0], axis=1))
#             temp_dict[num] = arr_ret10_re
#
#         total_num = int(self.backward_len / self.stride)
#         temp_list = [temp_dict[num] for num in range(1, total_num)]
#         temp_list.append(temp_dict[0])
#
#         result = np.concatenate(tuple(temp_list), axis=2)
#         return result
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], int(input_shape[2] / self.stride), input_shape[3]
