import tensorflow as tf
import numpy as np
import pandas as pd
import time
from LRP_linear_layer import *


class LSTM(object):
    def __init__(self, time_step, hidden_size, num_layers, learning_rate, batch_size, input_features):
        tf.reset_default_graph()
        self.time_step = time_step
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.input_features = input_features

        self.multi_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(num_units=n) for n in self.hidden_size])

        self.inputs = tf.placeholder(dtype=tf.float32, shape=[None, self.time_step, self.input_features])
        self.targets = tf.placeholder(dtype=tf.float32, shape=[None])
        self.states, self.states_c_h = tf.nn.dynamic_rnn(cell=self.multi_lstm_cell, inputs=self.inputs, dtype=tf.float32)
        self.states_last_h = self.states[:, -1, :]
        self.y_pre = tf.squeeze(tf.layers.dense(inputs=self.states_last_h, units=1, activation=None, use_bias=False), 1)
        self.loss = tf.reduce_mean(tf.square(self.targets - self.y_pre))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)  # 优化器
        self.train_op = optimizer.minimize(self.loss)

        self.saver = tf.train.Saver()  # Saver对象声明

    def get_a_cell(self):
        return tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=self.hidden_size)

    def train(self, x_train, y_train, n_epochs, save_path, log_every_n, early_stop):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            best = [100, 0]  # 记录loss和epoch
            earlystop = early_stop   # 早停止条件，若连续earlystop都没有出现更小的loss，则训练停止并保存最佳模型

            train_length = x_train.shape[0]
            split_index = round(train_length * 0.8)

            x_validation = x_train[split_index:]
            y_validation = y_train[split_index:]

            x_train = x_train[:split_index]
            y_train = y_train[:split_index]

            for epoch in range(n_epochs):
                index = np.arange(x_train.shape[0])
                np.random.shuffle(index)  # 训练集打乱数据顺序

                x_train = x_train[index]
                y_train = y_train[index]

                start = time.time()
                for iteration in range(x_train.shape[0] // self.batch_size + 1):
                    if iteration != x_train.shape[0] // self.batch_size:
                        x_batch = x_train[self.batch_size * iteration:self.batch_size * (iteration + 1)]
                        # print('x_batch', x_batch.shape)
                        label_batch = y_train[self.batch_size * iteration:self.batch_size * (iteration + 1)]
                    elif iteration == x_train.shape[0] // self.batch_size & x_train.shape[0] % self.batch_size != 0:
                        x_batch = x_train[self.batch_size * iteration:]
                        label_batch = y_train[self.batch_size * iteration:]
                    elif iteration == x_train.shape[0] // self.batch_size & x_train.shape[0] % self.batch_size == 0:
                        continue
                    feed = {self.inputs: x_batch, self.targets: label_batch}
                    loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed)
                    # cumulative_loss.append(loss)
                end = time.time()

                # 计算验证集上loss
                feed_validation = {self.inputs: x_validation, self.targets: y_validation}
                loss_validation, _ = sess.run([self.loss, self.train_op], feed_dict=feed_validation)

                if loss_validation < best[0]:
                    best = [loss_validation, epoch]
                    self.saver.save(sess, save_path + '/model', global_step=epoch)
                if epoch - best[1] > earlystop:  # 早停止条件
                    break

                if epoch % log_every_n == 0:
                    print('epoch: {}/{}... '.format(epoch, n_epochs),
                          'loss: {:.6f}... '.format(loss_validation),
                          '{:.4f} sec/epoch'.format((end - start)))

    def test(self, x_test, y_test, checkpoint_path):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path + '/'))
            # states, stacked_states, stacked_outputs = sess.run([self.states, self.stacked_states, self.stacked_outputs], feed_dict={self.inputs: x_test})
            # print(x_test.shape)
            y_hat_un = sess.run(self.y_pre, feed_dict={self.inputs: x_test})
            y_hat = y_hat_un
            y_labels = y_test
            return y_hat, y_labels

    def lrp(self, x_test, checkpoint_path, eps, bias_factor):
        with tf.Session() as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path + '/'))
            states, states_c_h = sess.run([self.states, self.states_c_h], feed_dict={self.inputs: x_test})
            y_hat_un = sess.run(self.y_pre, feed_dict={self.inputs: x_test})
            # print('x_test:', x_test.shape)

            # print('states.shape', states.shape)
            # print('states_c_h.shape', states_c_h[0][0].shape)

            ckpt = tf.train.get_checkpoint_state(checkpoint_path + '/')
            # print('ckpt:', ckpt.model_checkpoint_path)

            # 读取最新权重数据
            reader = tf.train.NewCheckpointReader(ckpt.model_checkpoint_path)
            # all_variables = reader.get_variable_to_shape_map()
            # print('all_variables:', all_variables)

            # 由于dynamic_rnn没有返回各时期的长期状态，要根据权重前向计算
            W_rnn = reader.get_tensor('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel')
            # print(W_rnn.shape)
            b_rnn = reader.get_tensor('rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias')
            # print(b_rnn.shape)
            W_dense = reader.get_tensor('dense/kernel')
            # print(W_dense.shape)

            T = x_test.shape[1]  # 时间步
            d = int(W_rnn.shape[-1] / 4)  # 神经网络隐藏层单元个数 要修改
            e = x_test.shape[2]  # 特征个数
            C = 1

            # gate indices (assuming the gate ordering in the LSTM weights is i,g,f,o):
            idx = np.hstack((np.arange(0, d), np.arange(2 * d, 4 * d))).astype(int)  # indices of gates i,f,o together
            idx_i, idx_g, idx_f, idx_o = np.arange(0, d), np.arange(d, 2 * d), np.arange(2 * d, 3 * d), np.arange(3 * d, 4 * d)

            h_states = np.zeros((T + 1, d))  # 各时间步隐含状态
            c_states = np.zeros((T + 1, d))  # 各时间步长期状态

            gates_x_h = np.zeros((T, 4 * d))
            gates_pre = np.zeros((T, 4 * d))
            gates_act = np.zeros((T, 4 * d))

            # for i in range(x_test.shape[0]):
            contribution = pd.DataFrame()
            for i in [0]:
                temp_x_test = x_test[i]
                # print(temp_x_test[0].shape)
                # print(h_states[0].shape)
                # 沿时间步展开
                for t in range(T):
                    # xt, ht-1拼接
                    x_h_concat = np.concatenate([temp_x_test[t], h_states[t-1]])
                    gates_x_h[t] = np.dot(W_rnn.T, x_h_concat)  # 4d * (e+d)和(e+d) * 1,得到4d*1
                    gates_pre[t] = gates_x_h[t] + b_rnn

                    gates_act[t, idx] = 1.0 / (1.0 + np.exp(- gates_pre[t, idx]))
                    gates_act[t, idx_g] = np.tanh(gates_pre[t, idx_g])

                    c_states[t] = gates_act[t, idx_f] * c_states[t - 1] + gates_act[t, idx_i] * gates_act[t, idx_g]
                    # h_states[t] = gates_act[t, idx_o] * np.tanh(c_states[t])
                    h_states[t] = states[0][t]
                y_dense = np.dot(W_dense.T, h_states[T - 1])[0]
                # print('隐藏状态：')
                # print('原始：', states[0])
                # print('前向计算：', h_states)
                # print('长期状态：')
                # print('原始：', states_c_h[0][0][0])
                # print('前向计算：', c_states)
                # print('预测值：')
                # print('原始：', y_hat_un[0])
                # print('前向计算：', y_dense)

                # LRP
                Rx = np.zeros(temp_x_test.shape)
                Rh_Left = np.zeros((T + 1, d))
                Rc_Left = np.zeros((T + 1, d))
                Rg_Left = np.zeros((T, d))  # gate g only
                Rout_mask = np.zeros((C))
                Rout_mask[0] = 1.0

                Rh_Left[T - 1] = lrp_linear(h_states[T - 1], W_dense, np.zeros((C)), np.array([y_hat_un[i]]), np.array([y_hat_un[i]]) * Rout_mask, d, eps, bias_factor, debug=False)
                for t in reversed(range(T)):
                    Rc_Left[t] += Rh_Left[t]
                    Rc_Left[t - 1] = lrp_linear(gates_act[t, idx_f] * c_states[t - 1], np.identity(d), np.zeros((d)), c_states[t], Rc_Left[t], d, eps, bias_factor, debug=False)
                    Rg_Left[t] = lrp_linear(gates_act[t, idx_i] * gates_act[t, idx_g], np.identity(d), np.zeros((d)), c_states[t], Rc_Left[t], d, eps, bias_factor, debug=False)
                    # self.Wxh_Left[idx_g].T
                    Rx[t] = lrp_linear(temp_x_test[t], W_rnn[np.arange(0, e)].T[idx_g].T, b_rnn[idx_g], gates_pre[t, idx_g], Rg_Left[t], d + e, eps, bias_factor, debug=False)
                    Rh_Left[t - 1] = lrp_linear(h_states[t - 1], W_rnn[np.arange(e, e + d)].T[idx_g].T, b_rnn[idx_g], gates_pre[t, idx_g], Rg_Left[t], d + e, eps, bias_factor, debug=False)
                # print('Rx_i.shape:', Rx_i.shape)
                contribution_i = np.sum(Rx, axis=0)
                contribution_i_df = pd.DataFrame(contribution_i)
                # print(contribution_i_df)
                contribution = pd.concat([contribution, contribution_i_df], axis=1, sort=True)
            return contribution

