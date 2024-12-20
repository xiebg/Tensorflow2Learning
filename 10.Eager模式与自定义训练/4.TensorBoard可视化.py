import tensorflow as tf
import datetime

# 1.创建数据集
mnist = tf.keras.datasets.mnist  # 手写数字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2.数据预处理
train_images = tf.expand_dims(train_images, axis=-1)  # 增加一个维度
test_images = tf.expand_dims(test_images, axis=-1)
train_images = tf.cast(train_images/255, tf.float32) # 归一化 到0-1之间
test_images = tf.cast(test_images/255, tf.float32)

train_labels  = tf.cast(train_labels, tf.int64)
test_labels = tf.cast(test_labels, tf.int64)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_dataset = train_dataset.repeat().shuffle(10000).batch(128)
test_dataset = test_dataset.repeat().batch(128)

# 3.创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(None,None,1)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 4.编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5.模型可视化
import os
log_dir = os.path.join("D:\Project\pythonProject\logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))   # 目录路径中不能存在中文
print(log_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1) # histogram_freq表示每隔多少个epoch保存一次直方图
# callbacks.EarlyStopping(monitor='val_loss', patience=3) # 当验证集的loss不再下降时，停止训练
# callbacks.LearRateScheduler(lambda epoch: 1e-3 * 10**(epoch / 20)) # 学习率调度器, 每20个epoch学习率乘以10的epoch/20次方

# 6.训练模型
model.fit(train_dataset,
          epochs=5,
          steps_per_epoch=60000//128,
          validation_data=test_dataset,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback]  # 传入回调函数
          )


# 启动TensorBoard
# tensorboard --logdir=D:\Project\pythonProject\logs\20241129-222115
# 打开浏览器，输入http://localhost:6006/