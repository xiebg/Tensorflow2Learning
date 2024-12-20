import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist  # 手写数字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 归一化数据
train_images = train_images / 255.0
test_images = test_images / 255.0

ds_train_img = tf.data.Dataset.from_tensor_slices(train_images)
ds_train_lab = tf.data.Dataset.from_tensor_slices(train_labels)
ds_train = tf.data.Dataset.zip((ds_train_img, ds_train_lab))
# ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

ds_train = ds_train.shuffle(10000).repeat().batch(64)  # 打乱前10000个数据，无限重复，批量为64
ds_test = ds_test.batch(64)
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

steps_per_epochs = train_images.shape[0] // 64 # 计算每个epoch的步数,//表示向下取整
validation_steps = test_images.shape[0] // 64
model.fit(ds_train,
          epochs=10,
          steps_per_epoch=steps_per_epochs, # steps_per_epoch表示每个训练集的epoch的步数，即每个epoch训练多少个batch
          validation_data=ds_test,
          validation_steps=validation_steps # validation_steps表示每个验证集的epoch的步数，即每个epoch验证多少个batch
          )
