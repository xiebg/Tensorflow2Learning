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
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

# （1）.创建写入器
file_writer = tf.summary.create_file_writer(log_dir+"/lr") # 创建写入器,写入日志文件到磁盘
file_writer.set_as_default() # 设置为默认写入器

# （2）.创建标量记录器
def lr_schedule(epoch):
    learning_rate = 0.1
    if epoch > 5:
        learning_rate = 0.01
    if epoch > 10:
        learning_rate = 0.001
    tf.summary.scalar(name='learning_rate', data=learning_rate, step=epoch) # （3）.记录学习率到日志文件中
    return learning_rate

# （4）.创建学习率调度器
lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule) # 创建学习率调度器

# 6.训练模型
model.fit(train_dataset,
          epochs=15,
          steps_per_epoch=60000//128,
          validation_data=test_dataset,
          validation_steps=10000//128,
          callbacks=[tensorboard_callback, lr_callback]  # （5）.传入回调函数
          )

# # （7）.启动tensorboard
# os.system(r"tensorboard --logdir=D:\Project\pythonProject\logs\20241129-225550")