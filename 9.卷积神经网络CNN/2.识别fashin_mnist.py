import numpy as np
import tensorflow as tf
# www.kaggle.com/训练模型
# 导入数据集
from matplotlib import pyplot as plt
def show_train_loss_info(history,name):
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend() # 显示图例
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.savefig(f'./img/{name}.png')
    plt.show()
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 数据预处理
# 扩张维度
train_images = np.expand_dims(train_images, axis=-1) # -1表示最后一个维度扩张
# 在最后一个维度上增加一个维度，表示通道数.将(60000, 28, 28)图像数据转换为4D张量，(60000, 28, 28, 1)
# (batch_size, height, width, channels)（批次大小，高度，宽度，通道数）
test_images = np.expand_dims(test_images, axis=-1)
# test_images = tf.expand_dims(test_images, axis=-1) # 也可以使用tf.expand_dims()函数,或者可以用reshape()函数

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=train_images.shape[1:],padding='same'),   # (28, 28, 1)
    tf.keras.layers.MaxPooling2D(),     # 池化层,默认池化窗口大小为(2,2),增加卷积层的视野范围（14，14，32）
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # (12, 12, 64)，填充模式为valid
    tf.keras.layers.GlobalAveragePooling2D(), # 全局平均池化层（将所有维度的特征图的平均值作为输出），将(None,12, 12, 64)转换为(None, 64)，减少参数数量。Flatten缺点是丢失了空间信息，并且引入了过多的计算参数
    tf.keras.layers.Dense(10, activation='softmax')
])
# model.output_shape # 输出形状 (None, 10) None表示batch_size

# 优化模型
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=train_images.shape[1:],padding='same'),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
show_train_loss_info(history,'fashion_mnist_cnn')

