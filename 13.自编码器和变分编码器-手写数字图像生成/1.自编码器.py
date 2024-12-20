import tensorflow as tf

# 自编码器实现：
# 1.搭建编码器
# 2.搭建解码器
# 3.设定损失函数

# 自编码器应用：
# 1.数据去噪
# 2.降维
# 3.图像生成

# 变分编码器： 【VAE，GAN(生成对抗网络)】-无监督学习
# 1.可以随机生成隐含变量
# 2.提高网络泛化能力，比普通自编码器好
# 3.缺点：生成的图像模糊
# 一方面让两张图片尽可能相似，另一方面让Code尽可能接近正态分布
# 隐含变量的限制方法：KL散度：衡量两个分布之间的差异，KL散度越小，两个分布越接近



# 手写数字MNIST数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)  # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x_train = x_train.reshape(x_train.shape[0], -1)  # (60000, 784) -1 代表其他维度自动计算
x_test = x_test.reshape(x_test.shape[0], -1)  # (10000, 784)

# 归一化
x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0

input_size = 784
hidden_size = 32
output_size = 784

inputs = tf.keras.layers.Input(shape=(input_size,))
# 编码器
en = tf.keras.layers.Dense(hidden_size, activation='relu')(inputs)
# 解码器
de = tf.keras.layers.Dense(output_size, activation='sigmoid')(en)

model = tf.keras.models.Model(inputs=inputs, outputs=de)
print(model.summary())

# 可视化模型
# tf.keras.utils.plot_model(model, to_file=r'.\img\encoder_model.png', show_shapes=True)
# 编译模型
model.compile(optimizer='adam',
              loss='mse'
              )
# 训练模型
model.fit(x_train, x_train,
          epochs=50,
          batch_size=256,
          shuffle=True,
          validation_data=(x_test, x_test))

# 获取编码器模型
encoder = tf.keras.models.Model(inputs=inputs, outputs=en)
# 获取解码器模型
input_de = tf.keras.layers.Input(shape=(hidden_size,))
output_de = model.layers[-1](input_de)
decoder = tf.keras.models.Model(inputs=input_de, outputs=output_de)

encode_test = encoder.predict(x_test[:10])
print(encode_test.shape)  # (10, 32)
decode_test = decoder.predict(encode_test)
print(decode_test.shape)  # (10, 784)
# 绘图
import matplotlib.pyplot as plt
x_test = x_test.numpy()
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(decode_test[i].reshape(28, 28))
plt.savefig(r'.\img\encoder_result.png')
plt.show()