import tensorflow as tf

# 解决深层网络的梯度消失或爆炸问题
# ResNet通过增加残差连接（shortcut connection,也叫shortcuts路径，即Residual Block）,显式地让网络中的层拟合残差映射(residual mapping)

# ResNet中所有Residual Block都没有pooling层，降采样是通过conv的stride实现的
# 通过Average Pooling层得到最终特征，而不是通过全连接层
# 每个卷积之后紧跟着Batch Normalization层，防止梯度消失或爆炸,Batch Normalization是对输入进行归一化，使得每层的输入分布变得更加稳定，从而加快收敛速度

class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        super(ResNetBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=3, padding='same',use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=3, padding='same',use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        shortcut = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = tf.nn.relu(out)
        out += shortcut     # 残差连接 shortcut connection
        out = tf.nn.relu(out)
        return out



resnet_block = ResNetBlock(64)

resnet50 = tf.keras.applications.ResNet50(weights=None,
                                          input_shape=(256,256,3))
print(resnet50.summary())

# 绘图
# tf.keras.utils.plot_model(resnet50, to_file='./image/resnet50.png', show_shapes=True)

