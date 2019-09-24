from datasets import Dataset
from model import LSTM
import numpy as np
import pandas as pd
import shutil
import os


def set_path(path):
    """创建或清空文件目录"""
    if os.path.exists(path) is False:
        os.makedirs(path)
    else:
        shutil.rmtree(path)


# LSTM参数
batch_size = 16  # 每批数据大小
time_step = 5  # 时间步长度
lstm_hidden_size = 10  # 隐藏层单元个数
input_features = 10  # 因子个数
learning_rate = 0.001  # 学习率
epochs = 30  # 模型训练轮数
early_stop = 40  # 早停止条件
log_every_n = 5  # 每训练多少轮打印信息

# LRP参数
eps = 0.001
bias_factor = 0.0


return_data = pd.read_excel('./data/monthly_return.xlsx', index_col=0)
stocks_code = list(return_data.columns)

print('-----------------------------------共有%d支个股-----------------------------------' % len(stocks_code))
for i in stocks_code:
    print('-------------------------当前目标股票代码：%s，为第%d支-------------------------' % (i, stocks_code.index(i) + 1))
    lstm_model_path = os.path.join('model/lstm', i)  # LSTM模型保存目录
    # set_path(lstm_model_path)

    dataset = Dataset(feature='./data/monthly_factor.xlsx', target='./data/monthly_return.xlsx', code=i)  # 读取数据

    # 得到训练集测试集数据
    x_train, x_test, y_train, y_test, test_basic_time = dataset.get_batch(time_step)
    _, scaler_for_y_l, _, _ = dataset.get_minmaxscaler()  # 得到归一化参数

    model_LSTM = LSTM(time_step=time_step, hidden_size=[lstm_hidden_size], num_layers=1, learning_rate=learning_rate, batch_size=batch_size, input_features=input_features)
    model_LSTM.train(x_train, y_train, epochs, lstm_model_path, log_every_n, early_stop)

    y_hat_arr_l, y_labels_arr_l = model_LSTM.test(x_test, y_test, lstm_model_path)
    print("LSTM模型%s测试集上均方根误差为%f：" % (i, np.sqrt(np.mean(np.square(y_hat_arr_l - y_labels_arr_l)))))
    # 反归一化
    y_pre_re_l = y_hat_arr_l * (scaler_for_y_l.data_max_ - scaler_for_y_l.data_min_) + scaler_for_y_l.data_min_
    y_test_labels_re_l = y_labels_arr_l * (
            scaler_for_y_l.data_max_ - scaler_for_y_l.data_min_) + scaler_for_y_l.data_min_
    test_basic_time.reset_index(drop=True, inplace=True)
    data = pd.concat([pd.DataFrame(test_basic_time), pd.DataFrame(y_pre_re_l), pd.DataFrame(y_test_labels_re_l)],
                     axis=1)
    data.columns = ['时间', '预测值', '真实值']
    data.to_csv('./data/prediction/' + i + '.csv', index=False, encoding='utf_8_sig')

    # LRP
    contribution = model_LSTM.lrp(x_test, lstm_model_path, eps, bias_factor)
    contribution_mean = contribution.mean(axis=1)
    print(contribution_mean.values)


