import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 手写数字MNIST数据集
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

# 模型架构
#    1.编码器--->μ,σ（均值和标准差）
#    2.μ,σ--->隐含变量
#    3.隐含变量--->解码器--->重构图像

# 数据标准化
train_images = train_images.reshape(train_images.shape[0], -1)  # (60000, 784) -1 代表其他维度自动计算
test_images = test_images.reshape(test_images.shape[0], -1)  # (10000, 784)
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.
dataset_train = tf.data.Dataset.from_tensor_slices(train_images)
dataset_test = tf.data.Dataset.from_tensor_slices(test_images)
dataset = dataset_train.shuffle(60000).batch(128)
dataset_test = dataset_test.batch(128)

# 编码器
class VAE_model(tf.keras.Model):
    def __init__(self):
        super(VAE_model, self).__init__()
        self.linear1 = tf.keras.layers.Dense(400)
        self.linear2 = tf.keras.layers.Dense(20)  # 输出μ*20
        self.linear3 = tf.keras.layers.Dense(20)  # 输出σ*20

        self.linear4 = tf.keras.layers.Dense(400)
        self.linear5 = tf.keras.layers.Dense(784)  # 输出重构图像

    def encode(self, x):  # 编码器
        h1 = tf.nn.relu(self.linear1(x))
        return self.linear2(h1), self.linear3(h1)  # μ, σ方差的对数值

    def reparameterize(self, mu, logvar): # 重参数化
        std = tf.exp(0.5 * logvar)  # 方差
        eps = tf.random.normal(std.shape)  # 正态分布
        return mu + eps * std  # 重参数化

    def decode(self, z):  # 解码器
        h4 = tf.nn.relu(self.linear4(z))
        return tf.nn.sigmoid(self.linear5(h4))  # 重构图像

    def call(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)  # 重参数化(隐含变量)
        return self.decode(z), mu, logvar  # 重构图像, μ, σ方差的对数值



def loss_func(recon_x, x, mu, logvar):
    BCE_loss = tf.keras.losses.binary_crossentropy(x, recon_x) # 交叉熵损失,逻辑回归的损失函数
    KLD_loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))  # KLD散度
    a = 0.001  # 超参数
    return BCE_loss + a*KLD_loss
    # return tf.reduce_mean(BCE_loss + KLD_loss)  # 平均损失

# 实例化模型
model = VAE_model()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
epoch_loss_avg = tf.keras.metrics.Mean("train_loss") # 记录每个epoch的loss


def train_step(model, images):
    with tf.GradientTape() as tape:  # 记录梯度信息
        pred_img,mu,logvar = model(images)  # 前向传播
        loss_value = loss_func(pred_img, images, mu, logvar)  # 计算损失
    grads = tape.gradient(loss_value, model.trainable_variables)  # 计算梯度
    opt.apply_gradients(zip(grads, model.trainable_variables))  # 更新参数
    epoch_loss_avg(loss_value)  # 记录loss
    return pred_img

def train():
    train_loss_results = []
    num_epochs = 330

    for epoch in range(num_epochs):
        for images in dataset:
            pred_img = train_step(model, images)
        print(epoch)

        train_loss_results.append(epoch_loss_avg.result())
        epoch_loss_avg.reset_states()  # 重置loss
        print("Epoch {:03d}: Loss: {:.3f}".format(epoch+1, train_loss_results[-1]))

        for i in range(64):
            plt.subplot(8, 8, i+1)
            plt.imshow(pred_img[i].numpy().reshape(28, 28), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
        plt.show()

train()

# 应用变分自编码器进行图像压缩
# 随机生成logvar和mu交给reparameterize函数，生成隐含变量z，再通过解码器生成重构图像。
# 生成100张图片
test_mu = np.linspace(-0.9, 0.9, 2000).reshape(100, 20).astype('float32') #np.linspace(-0.9, 0.9, 2000)生成2000个均匀分布的数
test_logvar = np.linspace(-0.9, 0.9, 2000).reshape(100, 20).astype('float32')

test_img = model.decode(model.reparameterize(test_mu, test_logvar))
print(test_img.shape)   # (100, 784)
test_img = test_img.numpy().reshape(100, 28, 28)

plt.figure(figsize=(6, 6))
for i in range(100):
    plt.subplot(10, 10, i+1)
    plt.imshow(test_img[i], cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
plt.show()