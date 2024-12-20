import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集 C:\Users\retur\.keras\datasets 自动下载
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# show_image_info(train_images, train_labels)
# 归一化数据
train_images = train_images / 255.0
test_images = test_images / 255.0

def show_image_info(train_images, train_labels):
    # 查看数据集信息
    print(train_images.shape)
    print(train_labels.shape)
    # 画出25张图片在一张图上
    plt.figure(figsize=(10, 10))  # 设置画布大小
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(train_labels[i])
    plt.show()
    print("图片最大值：", np.max(train_images[0]))
    print(train_labels)  # 9代表帽子，0代表T恤，2代表连衣裙，3代表外套，4代表裤子，5代表鞋子，6代表包，7代表衬衫，8代表运动鞋


def show_train_loss_info(history,name):
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend() # 显示图例
    plt.savefig(f'./img/{name}.png')
    plt.show()


def show_train_accuracy_info(history,name):
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.legend() # 显示图例
    plt.savefig(f'./img/{name}.png')
    plt.show()



def train_model(train_images, train_labels, test_images, test_labels):
    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # 展平28*28的图片 输入层 Flatten层表示多维数据扁平化
        tf.keras.layers.Dense(128, activation='relu'),  # 全连接层 128个神经元，激活函数relu
        tf.keras.layers.Dense(10, activation='softmax')  # 全连接层 10个神经元，输出层 softmax表示把10个输出变成概率分布
    ])

    model.compile(optimizer='adam', # optimizer=tf.keras.optimizers.Adam(lr=0.001)
                  loss='sparse_categorical_crossentropy',
                  # 交叉熵损失函数 sparse_categorical_crossentropy适用于多分类问题,处理数字标签； categorical_crossentropy适用于多分类问题，处理one-hot编码标签
                  metrics=['accuracy'])

    # 训练模型
    model.fit(train_images, train_labels, epochs=5)

    # 评估模型
    test_loss, test_acc = model.evaluate(test_images, test_labels) # evaluate()方法用于评估模型在测试数据上的性能 predict（）方法用于预测模型在测试数据上的输出
    print('测试准确率:', test_acc)

    # 预测模型
    predictions = model.predict(test_images[0:10])
    show_image_info(test_images, test_labels)
    print(predictions)
    print(np.argmax(predictions, axis=1))

def train_model_one_hot(train_images, train_labels, test_images, test_labels):
    tarin_labels_one_hot = tf.keras.utils.to_categorical(train_labels, num_classes=10) # 将标签转换为one-hot编码, num_classes=10可以省略 [0,1,0,0,0,0,0,0,0,0]
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, num_classes=10) # 将标签转换为one-hot编码

    # 构建模型
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # 展平28*28的图片 输入层 Flatten层表示多维数据扁平化
        tf.keras.layers.Dense(128, activation='relu'),  # 全连接层 128个神经元，激活函数relu
        # 解决过拟合问题，添加Dropout层
        tf.keras.layers.Dropout(0.5), # rate=0.5 表示随机丢弃50%的神经元

        tf.keras.layers.Dense(128, activation='relu'),
        # 解决过拟合问题，添加Dropout层
        tf.keras.layers.Dropout(0.5),  # rate=0.5 表示随机丢弃50%的神经元

        tf.keras.layers.Dense(128, activation='relu'),
        # 解决过拟合问题，添加Dropout层
        tf.keras.layers.Dropout(0.2),  # rate=0.2 表示随机丢弃20%的神经元

        tf.keras.layers.Dense(10, activation='softmax')  # 全连接层 10个神经元，输出层 softmax表示把10个输出变成概率分布
    ])

    model.compile(optimizer='adam', # adma(adaptive moment estimation)优化器,中文名：自适应矩估计
                  loss='categorical_crossentropy', # categorical_crossentropy适用于多分类问题，处理one-hot编码标签
                  metrics=['accuracy']) # metrics=['accuracy']表示在训练过程中，会计算准确率

    # 训练模型
    history = model.fit(train_images, tarin_labels_one_hot, epochs=10,
              validation_data=(test_images, test_labels_one_hot) # validation_data用于指定验证集数据，在训练过程中，会定期计算验证集的损失和指标，以帮助判断模型是否过拟合
              )

    show_train_loss_info(history, "dropout_过拟合_loss")
    show_train_accuracy_info(history, "dropout_过拟合_accuracy")


    # 预测模型
    predictions = model.predict(test_images[0:10])
    print(predictions.shape) # (10, 10) 10个样本，每个样本有10个概率
    print(np.argmax(predictions, axis=1)) # 每个样本有10个概率，每个概率代表对应类别的概率 [9 2 1 1 6 1 4 6 5 7]


# train_model(train_images, train_labels, test_images, test_labels)
train_model_one_hot(train_images, train_labels, test_images, test_labels)