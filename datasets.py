import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, feature, target, code):
        self.code = code
        self.x_data = pd.DataFrame()
        self.y_data = pd.Series()
        self.basic_time = pd.DataFrame()
        self.load_data(feature, target)

    def load_data(self, feature, target):
        target_data = pd.read_excel(target, index_col=0)
        factor_data = pd.read_excel(feature, None)
        for f in list(factor_data.keys()):
            self.x_data = pd.concat([self.x_data, factor_data[f].loc[:, self.code]], axis=1, sort=True)
        self.basic_time = pd.DataFrame(target_data.index)
        self.y_data = target_data[1:][self.code]
        self.x_data.columns = list(factor_data.keys())
        self.x_data = self.x_data[:-1]

    def get_minmaxscaler(self):
        scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 按列做minmax缩放
        scaler_for_y = MinMaxScaler(feature_range=(0, 1))  # 按列做minmax缩放
        if pd.DataFrame(self.x_data).shape[-1] == 1:
            scaled_x_data = scaler_for_x.fit_transform(self.x_data.to_frame())
        else:
            scaled_x_data = scaler_for_x.fit_transform(self.x_data)
        scaled_y_data = scaler_for_y.fit_transform(self.y_data.to_frame())
        return scaler_for_x, scaler_for_y, scaled_x_data, scaled_y_data

    def get_batch(self, n_steps):
        _, _, scaled_x_data, scaled_y_data = self.get_minmaxscaler()

        x_batch = np.empty(shape=[scaled_x_data.shape[0] - n_steps + 1, n_steps, scaled_x_data.shape[-1]])
        y_batch = np.empty(shape=[scaled_x_data.shape[0] - n_steps + 1])

        t = 0  # 定义游标 每次加1
        # print(scaled_x_data.shape[0])
        for i in range(scaled_x_data.shape[0] - n_steps + 1):
            temp_x = scaled_x_data[t:t + n_steps, :]
            temp_y = scaled_y_data[t + n_steps - 1, :]
            x_batch[i, :] = temp_x
            y_batch[i] = temp_y
            t += 1
        x_train, x_test, y_train, y_test = train_test_split(x_batch, y_batch, test_size=0.274, random_state=0,
                                                            shuffle=False)
        # print()
        return x_train, x_test, y_train, y_test, self.basic_time[self.y_data.shape[0] - x_test.shape[0] + 1:]
