import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# 导入电影评论数据集
data = keras.datasets.imdb
(x_train, y_train), (x_test, y_test) = data.load_data(num_words=10000) # num_words参数指定保留的最大词汇数
print("Testing entries: {}, labels: {}".format(x_test.shape, y_test.shape))
# print(x_train[0]) # 打印第一条评论 整数序列
# print(data.get_word_index()) # 打印词汇表
# 编码评论数据
# 1. tf.idf 编码,传统方法
# 2. word2vec编码,将每个单词映射为一个固定长度的向量，文本训练成密集向量
# 3. k-hot编码,将每个单词用一个向量表示,向量的第i位为1表示单词i出现过,0表示单词i未出现过
# 4. 卷积神经网络编码,将整个评论视为一个序列,使用卷积神经网络处理序列数据

# 文本训练成密集向量
# print([len(x) for x in x_train]) # 打印每条评论的长度
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=300) # 填充或截断每条评论使其长度为300
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=300)

# text = "I love to eat apples" # 测试用例
# text_dict = dict((word,text.split().index(word)) for word in text.split()) # 词汇表
# print(text_dict) # 打印词汇表

# 构建模型
model = keras.Sequential()
model.add(layers.Embedding(10000, 50, input_length=300)) # 将每个单词映射为一个16维的向量,输入序列长度为300,50为输出维度
# 经过上面这一层,输入数据为从（25000，300）变为（25000，300，50）
model.add(layers.GlobalAveragePooling1D()) # 将（25000，300，50）变为（25000，300*50）
model.add(layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.01))) # 全连接层 kernel_regularizer参数用于控制权重的正则化
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid')) # 输出层,二分类问题,1为好评,0为差评
# 编译模型
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # 二分类问题,使用binary_crossentropy损失函数
    metrics=['accuracy'])

# model.summary() # 打印模型结构
# 训练模型
history = model.fit(x_train, y_train,
                    epochs=15, # 训练轮数
                    batch_size=256, # 批处理大小
                    validation_data=(x_test, y_test) # 验证集
                    )
def show_train_accuracy_info(history,name):
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.plot(history.epoch, history.history['accuracy'],"r",label='accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'],"b--",label='val_accuracy')
    plt.legend() # 显示图例
    plt.savefig(f'./img/{name}.png')
    plt.show()
show_train_accuracy_info(history,'text_dropout_accuracy')

# 解决过拟合问题
# 1. 增加dropout层
# 2. 正则化 l1, l2正则化, lasso, ridge 回归

