import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 归一化数据
train_images = train_images / 255.0
test_images = test_images / 255.0

# 构建函数式模型网络
input_data = tf.keras.Input(shape=(28,28,1))
x = tf.keras.layers.Flatten()(input_data)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output_data = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=input_data, outputs=output_data)
model.summary()

input_data1 = tf.keras.Input(shape=(28,28,1))
input_data2 = tf.keras.Input(shape=(28,28,1))
x1 = tf.keras.layers.Flatten()(input_data1)
x2 = tf.keras.layers.Flatten()(input_data2)
x = tf.keras.layers.concatenate([x1, x2]) # 合并特征
# ...... 省略中间层

model = tf.keras.Model(inputs=[input_data1, input_data2], outputs=output_data)
model.summary()