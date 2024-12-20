
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# www.kaggle.com/c/leaf-classification/overview
# 读取数据
data = pd.read_csv('./data/leaf/train.csv')
# 99个物种，各16个样本
# print(data.species.unique())
print(len(data.species.unique()))  # 99
print(data.shape)  # (990, 194)
labels_tuple = pd.factorize(data.species)  # 将字符串标签转换为数字标签
# print(labels_tuple)  # [0 1 2 ... 55,  7, 13] ['Acer_Opalus', 'Pterocarya_Stenoptera',...,'Sorbus_Aria']
labels = labels_tuple[0]
x = data[data.columns[2:]]
# 划分训练集和测试集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, labels,
                                                    # test_size=0.2, random_state=1  # (776, 16) (144, 16) (776,) (144,)
                                                    )
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (742, 192) (248, 192) (742,) (248,)
# 标准化模型输入
mean = x_train.mean(axis=0)  # 按列求均值
std = x_train.std(axis=0)  # 按列求标准差
x_train = (x_train - mean) / std
test_x = (x_test - mean) / std

print(x_train.shape)  # (742, 192)
# 把每条数据看成一个序列，输入到卷积层
# 一维卷积和Lstm都需要（samples, timesteps, features）的输入格式,分别（对应样本数，时间步数，特征数）
train_x = np.expand_dims(x_train, axis=-1)
# print(train_x.shape)  # (742, 192, 1)
test_x = np.expand_dims(test_x, axis=-1)
# print(test_x.shape)  # (248, 192, 1)

def show_train_loss_info(history,name):
    import os
    if not os.path.exists('./image'):
        os.mkdir('./image')
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend() # 显示图例
    plt.savefig(f'./image/{name}.png')
    plt.show()
def show_train_accuracy_info(history,name):
    import os
    if not os.path.exists('./image'):
        os.mkdir('./image')
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.legend() # 显示图例
    plt.savefig(f'./image/{name}.png')
    plt.show()

# 定义模型
def simple_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 7, activation='relu', input_shape=(192, 1), padding='same'))
    model.add(tf.keras.layers.Conv1D(32, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dense(99, activation='softmax'))
    return model
def simple_model1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 7, activation='relu', input_shape=(192, 1), padding='same'))
    model.add(tf.keras.layers.Conv1D(32, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(64, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(128, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(256, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(256, 7, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(99, activation='softmax'))
    return model

def simple_model2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(32, 11, activation='relu', input_shape=(192, 1), padding='same'))
    model.add(tf.keras.layers.Conv1D(32, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(64, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(64, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(128, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(128, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Conv1D(256, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.Conv1D(256, 11, activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling1D(3))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.GlobalMaxPooling1D())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(99, activation='softmax'))
    return model
# model = simple_model()
# model = simple_model1()
model = simple_model2()
print(model.summary())
model.compile(optimizer='RMSprop', # RMSprop擅长处理序列问题
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_x, y_train,
                    epochs=400,
                    # batch_size=128,
                    validation_data=(test_x, y_test))

show_train_loss_info(history,'leaf_train_loss2')
show_train_accuracy_info(history,'leaf_train_accuracy2')

