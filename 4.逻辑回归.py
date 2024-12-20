import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('./data/credit-a.csv', header=None)

y_counta = data.iloc[:, -1].value_counts()  # 统计标签数量
print(y_counta)

# 数据预处理
x = data.iloc[:, :-1]
y = data.iloc[:, -1].replace(-1, 0)


def train_model(x, y):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, input_shape=(15,), activation='relu'),  # 输入层 4个神经元
        tf.keras.layers.Dense(4, activation='relu'),  # 没有input_shape 则默认计算输入维度
        tf.keras.layers.Dense(1, activation='sigmoid')  # 输出层 1个神经元 输出概率
    ])

    print(model.summary())

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # 交叉熵损失函数
                  metrics=['accuracy'])  # 准确率评估, 评估指标 accuracy是默认的（可以写成acc），也可以指定其他评估指标，如AUC、F1-score等

    # 训练模型
    # history = model.fit(x, y, epochs=100, batch_size=32, validation_split=0.2) # 训练模型，指定训练轮数、批次大小、验证集比例
    history = model.fit(x, y, epochs=100)  # 训练模型
    print(history)

    # # 保存模型
    # model.save('./model/credit-a.h5')

    return model, history


def plot_history(history):
    print(history.history.keys())  # dict_keys(['loss', 'accuracy'])
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.show()
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    plt.show()


model, history = train_model(x, y)
# plot_history(history)

# # 加载模型
# model_loaded = tf.keras.models.load_model('./model/credit-a.h5')

# 预测结果
y_pred = model.predict(x.iloc[:10,:])  # 预测前10个样本的结果
print(y_pred)# 大于0.5的概率为正样本概率，小于0.5的概率为负样本概率
