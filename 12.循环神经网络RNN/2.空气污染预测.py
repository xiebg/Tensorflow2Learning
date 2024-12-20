import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# 读取数据集
data = pd.read_csv('./data/beijing_air.csv')

# 处理NaN值
# print(data['pm2.5'].isnull().sum()) # 2067
# 去掉前面24条NaN值,其他的填充
data = data.iloc[24:].fillna(method='ffill')  # ffill:用前一个值填充

# 处理时间序列
# print(data.columns.values) # ['No' 'year' 'month' 'day' 'hour' 'pm2.5' 'DEWP' 'TEMP' 'PRES' 'cbwd''Iws' 'Is' 'Ir']
# 时间合并为日期
# data['date'] = pd.to_datetime(data[['year','month', 'day', 'hour']])
data['date'] = data.apply(lambda x: datetime.datetime(x['year'], x['month'], x['day'], x['hour']), axis=1)
# print(data.head())

# 将时间序列设置为index
data.drop(['No', 'year', 'month', 'day', 'hour'], axis=1, inplace=True)
data.set_index('date', inplace=True)
# print(data.head())

# print(data.cbwd.unique()) # ['SE' 'cv' 'NW' 'NE']
data = data.join(pd.get_dummies(data.cbwd, prefix='cbwd'))  # 增加cbwd的one-hot编码,prefix:列名前缀
del data['cbwd']
print(data.columns.values)  # ['pm2.5' 'DEWP' 'TEMP' 'PRES' 'Iws' 'Is' 'Ir' 'cbwd_NE' 'cbwd_NW' 'cbwd_SE' 'cbwd_cv']
print(data.head())

# # 最后1000个pm2.5的折线图
# data['pm2.5'][-1000:].plot(figsize=(12, 8))
# plt.show()
# data['TEMP'][-1000:].plot(figsize=(12, 8))
# plt.show()

seq_length = 5 * 24  # 观察前面5天的数据
delay = 1 * 24  # 预测后1天的数据

# 训练数据是前面5*24个数据,目标数据是1*24个数据后面的数据

# 六天为一组
data_ = []
for i in range(len(data) - seq_length - delay):
    data_.append(data[i:i + seq_length + delay])

print(len(data_))  # 43656
print(data_[0].shape)  # (144, 11)
# 转换为numpy数组
data_ = np.array([df.values for df in data_])

data_ = np.array(data_)
print(data_.shape)  # (43656, 144, 11)

# 乱序
np.random.shuffle(data_)

# 取每组前面的5*24个数据作为输入,后面的最后1个数据的pm2.5(第一列)作为输出
x = data_[:, :seq_length, :]  # [batch,行数,列数]
y = data_[:, -1:, 0]

# 划分训练集和测试集
train_size = int(data_.shape[0] * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 数据标准化
mean = x_train.mean(axis=0)  # 计算每列的均值
std = x_train.std(axis=0)  # 计算每列的标准差
x_train = (x_train - mean) / std  # 标准化

# x_test = (x_test - x_test.mean(axis=0)) / x_test.std(axis=0)  # 标准化,训练的时候不可能知道测试集的均值和标准差
x_test = (x_test - mean) / std  # 标准化

def show_train_loss_info(history, name):
    import os
    if not os.path.exists('./image'):
        os.mkdir('./image')
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend()  # 显示图例
    plt.savefig(f'./image/{name}.png')
    plt.show()


def show_train_mae_info(history, name):
    import os
    if not os.path.exists('./image'):
        os.mkdir('./image')
    plt.plot(history.epoch, history.history['mae'], label='mae')
    plt.plot(history.epoch, history.history['val_mae'], label='val_mae')
    plt.legend()  # 显示图例
    plt.savefig(f'./image/{name}.png')
    plt.show()


# 构建模型

def simple_model():
    # 模型从简单到复杂:
    batch_size = 128
    model = tf.keras.Sequential()
    # 接收二维数据
    model.add(tf.keras.layers.Flatten(input_shape=(x_train.shape[1:])))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])  # mae:平均绝对误差

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=100,  # validation_split=0.2, # 验证集
                        validation_data=(x_test, y_test))

    show_train_loss_info(history, 'train_loss_dense')
    show_train_mae_info(history, 'train_mea_dense')


# simple_model()

def simple_lstm_model():
    batch_size = 128
    model = tf.keras.Sequential()
    # 接收三维数据
    model.add(tf.keras.layers.LSTM(32, input_shape=(120,11)))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])  # mae:平均绝对误差

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=100,  # validation_split=0.2, # 验证集
                        validation_data=(x_test, y_test))

    show_train_loss_info(history, 'train_loss_lstm')
    show_train_mae_info(history, 'train_mea_lstm')


# simple_lstm_model()

model_path = r'D:\Project\pythonProject\models\lstm_model.h5'
def lstm_model():  # 多层LSTM
    batch_size = 128
    model = tf.keras.Sequential()
    # 接收三维数据
    model.add(tf.keras.layers.LSTM(64,
                                   input_shape=(x_train.shape[1:]),
                                   return_sequences=True))  # return_sequences=True:返回所有时间步的输出,保证输出的数据为三维数据
    model.add(tf.keras.layers.LSTM(32, return_sequences=True))
    model.add(tf.keras.layers.LSTM(16, return_sequences=True))
    model.add(tf.keras.layers.LSTM(8))
    model.add(tf.keras.layers.Dense(1))

    # 在训练过程中降低学习率
    lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',  # 监控值
                                         patience=3,  # 当patience个epoch没有提升时,学习率衰减
                                         factor=0.5,  # 学习率衰减时的衰减比例
                                         min_lr=1e-6,  # 学习率下限
                                         verbose=1  # 显示信息
                                         )

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['mae'])  # mae:平均绝对误差

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=100,
                        callbacks=[lr_reduce],  # 回调函数
                        validation_data=(x_test, y_test))
    show_train_loss_info(history, 'train_loss_lstm2')
    show_train_mae_info(history, 'train_mea_lstm2')

    model.save(model_path)  # 保存模型

# lstm_model()

# 使用模型进行预测
# 加载模型
model = tf.keras.models.load_model(model_path)

model.evaluate(x_test, y_test, verbose=0)  # 评估模型 verbose=2:显示详细信息,1显示进度条，0不显示
# 预测
# 对于多条数据预测
y_pred = model.predict(x_test)   # 返回ndarray类型
print(y_pred.shape)  # (8732, 1)
# 对于一条数据预测
data_test = data[-120:]
# 数据标准化
data_test = (data_test - mean) / std # 一定是训练时的均值和标准差
data_test = data_test.to_numpy()
print(data_test.shape)  # (120, 11)
# data_test = data_test.reshape(1, 120, 11)
data_test = np.expand_dims(data_test, axis=0)  # (1, 120, 11)
print(data_test.shape)  # (1, 120, 11)
y_pred = model.predict(data_test)
print(y_pred)  # 预测值