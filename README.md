# LSTM_LRP_factor
使用LSTM及股票因子数据预测未来收益，使用LRP(layer-wise relevance propagation)增强网络可解释性
# 模型结构
包括数据预处理、模型训练测试及LRP反向传播三个部分
# 代码框架
preprocessing.py：因子数据预处理文件；</br>
main.py：主模型入口；</br>
datasets.py：特征向量处理；</br>
model.py：LSTM模型，包含模型训练、测试及LRP；</br>
LRP_linear_layer.py：[LRP计算公式](https://github.com/ArrasL/LRP_for_LSTM/blob/master/code/LSTM/LRP_linear_layer.py)；</br>
utils.py：基本函数；</br>
