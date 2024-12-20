from tensorflow import keras
import tensorflow as tf

tf.keras.layers.Conv2D(
    filters=32, # 卷积核数量
    kernel_size=3, # 卷积核大小 3x3,一般可以选用3x3,5x5
    strides=1, # 步长
    padding='same', # 填充方式 same 表示输出大小与输入大小相同 valid 表示不填充,默认valid
    activation='relu', # 激活函数
    kernel_initializer='he_normal', # 卷积核初始化方式,默认glorot_uniform,
    # he_normal是针对relu激活函数的初始化方式,glorot_uniform是针对sigmoid激活函数的初始化方式
    # glorot_uniform表示权重服从均匀分布,he_normal表示权重服从高斯分布，目前效果更好的初始化方式是glorot_uniform
    kernel_regularizer=None, # 正则化项
    input_shape=(28, 28, 1) # 输入形状
)
tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), # 池化核大小
    strides=2, # 步长
    padding='same' # 填充方式,默认为valid
)