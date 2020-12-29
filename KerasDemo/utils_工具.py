from tensorflow.keras.utils import normalize
import pandas as pd

x_data = pd.read_pickle(r'D:\code\AlphaNet\db\2020-10-30\xdata_b30_f10.pkl')
x_data_n = normalize(x_data,axis=0)
print(x_data)
print(x_data_n)