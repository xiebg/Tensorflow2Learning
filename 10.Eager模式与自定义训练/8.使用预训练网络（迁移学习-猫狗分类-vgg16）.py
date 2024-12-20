import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 加载数据集
train_image_paths = glob.glob('.\\data\\dc_2000\\train\\*\\*.jpg')
# cat 为 1，dog 为 0
train_image_labels = [int(a_image_paths.split('\\')[-2] == 'cat') for a_image_paths in train_image_paths]

test_image_paths = glob.glob('.\\data\\dc_2000\\test\\*\\*.jpg')
test_image_labels = [int(a_image_paths.split('\\')[-2] == 'cat') for a_image_paths in test_image_paths]

# 定义数据集加载函数
def load_train_image(img_path,label,channels=3):
    img_raw = tf.io.read_file(img_path) # 读取图片 二进制格式
    img_tensor = tf.image.decode_jpeg(img_raw,channels=channels) # 解码图片 channels=3表示RGB三通道 decode_image()是通用的
    img_tensor = tf.image.resize(img_tensor, [256, 256]) # 调整图片大小为256x256
    img_tensor = tf.cast(img_tensor, tf.float32) # 将uint8转换为float32
    # # tf.image.convert_image_dtype() 如果输入是uint8，则输出是float32，并且会归一化，如果是float则不会归一化
    img_tensor = img_tensor/255.0 # 归一化到0-1之间
    label = tf.reshape(label, [1]) # 转换为1维张量
    return img_tensor,label

# 定义数据集加载函数，数据增强
def load_train_image_aug(img_path,label,channels=3):
    img_raw = tf.io.read_file(img_path) # 读取图片 二进制格式
    img_tensor = tf.image.decode_jpeg(img_raw,channels=channels) # 解码图片 channels=3表示RGB三通道 decode_image()是通用的
    img_tensor = tf.image.resize(img_tensor, [360, 360]) # 调整图片大小为360x360,通道可以省略
    # 数据增强
    img_tensor = tf.image.random_crop(img_tensor, [256, 256, 3]) # 随机裁剪为256x256x3,通道不可以省略
    img_tensor = tf.image.random_flip_left_right(img_tensor) # 随机左右翻转
    img_tensor = tf.image.random_flip_up_down(img_tensor) # 随机上下翻转
    img_tensor = tf.image.random_brightness(img_tensor, 0.2) # 随机调整亮度，效果不明显
    img_tensor = tf.image.random_contrast(img_tensor, 0.8, 1.2) # 随机调整对比度，效果不明显

    img_tensor = tf.cast(img_tensor, tf.float32) # 将uint8转换为float32
    img_tensor = img_tensor/255.0 # 归一化到0-1之间
    label = tf.reshape(label, [1]) # 转换为1维张量
    return img_tensor,label

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
AUTOTUNE = tf.data.experimental.AUTOTUNE # 自动选择加速计算的方式
train_dataset = train_dataset.map(load_train_image, num_parallel_calls=AUTOTUNE) # 加载图片并预处理
Batch_size = 32
train_count = len(train_image_paths)
train_dataset = train_dataset.shuffle(train_count).batch(Batch_size) # 打乱数据集并分批
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE) # 预取数据集

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
test_dataset = test_dataset.map(load_train_image, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.batch(Batch_size)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
test_count = len(test_image_paths)

# 定义模型
# keras中内置的vgg16模型
# 预训练的卷积基层
covn_base = tf.keras.applications.VGG16(
    weights='imagenet', # weights='imagenet', # 加载预训练模型，imagenet上预训练的模型
    include_top=False,  # include_top=False, # 不包含顶层的全连接层和softmax层，只有卷积基
    input_shape=(256, 256, 3))
convn_base2 = tf.keras.applications.xception.Xception( # Xception网络模型
    weights='imagenet',# 加载预训练模型，imagenet上预训练的模型
    include_top=False,  # 不包含顶层的全连接层和softmax层，只有卷积基
    input_shape=(256, 256, 3), # 输入图片的大小
    pooling='avg') # 池化方式，avg表示全局平均池化，max表示全局最大池化

covn_base.trainable = False # 冻结预训练模型的卷积基层
model = tf.keras.Sequential()
model.add(covn_base) # 添加预训练的卷积基层
model.add(tf.keras.layers.GlobalAveragePooling2D()) # 添加全局平均池化层
model.add(tf.keras.layers.Dense(512, activation='relu')) # 添加全连接层
model.add(tf.keras.layers.BatchNormalization()) # 添加批归一化层
model.add(tf.keras.layers.Dropout(0.5))   # 添加dropout层
model.add(tf.keras.layers.Dense(1, activation='sigmoid')) # 添加输出层

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 训练模型
history = model.fit(
    train_dataset,
    steps_per_epoch=train_count//Batch_size, # 训练集的步数
    epochs=15, # 训练的轮数
    validation_data=test_dataset, # 验证集
    validation_steps=test_count//Batch_size # 验证集的步数
)


# 微调模型，微调最后三层，必须前面训练好才可以训练这里
covn_base.trainable = True # 解冻预训练模型的卷积基层
print(len(covn_base.layers)) # 打印卷积基层的层数
fine_tune_at = -3 # 微调的层数,最后三层
for layer in covn_base.layers[:fine_tune_at]:
    layer.trainable =  False # 冻结前面的层

# 重新编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0005/10), # 学习率调低
              loss='binary_crossentropy',
              metrics=['accuracy'])

initial_epochs = 15 # 初始训练的轮数
fine_tune_epochs = 10 # 微调的轮数
total_epochs = initial_epochs + fine_tune_epochs # 总的轮数
# 重新训练模型
history_fine = model.fit(
    train_dataset,
    steps_per_epoch=train_count//Batch_size, # 训练集的步数
    epochs=total_epochs, # 训练的总轮数
    initial_epoch=initial_epochs, # 初始轮数（已经训练过的轮数）
    validation_data=test_dataset, # 验证集
    validation_steps=test_count//Batch_size # 验证集的步数
)

Xception_base = tf.keras.applications.xception.Xception(
    weights='imagenet', # weights='imagenet', # 加载预训练模型，imagenet上预训练的模型，None表示随机初始化
    include_top=False,  # 不包含顶层的全连接层和softmax层，只有卷积基,默认为True
    input_tensor=None,  # 输入张量，如果为None，则自动创建
    input_shape=(299, 299, 3), # 输入张量的形状，支支持（高度，宽度，通道数）的维度顺序，默认输入尺寸为299x299x3,仅当include_top=False时有效，否则输入形状必须为299x299x3，宽高必须不小于71
    pooling=None,  # 全局池化方式，如果为None，则不池化,直接输出最后一个卷积层的输出，是一个4D张量，shape为(batch_size, height, width, channels)
    # "avg" 代表全局平均池化，ClobalAveragePooling2D,相当于在最后一个卷积层后面加了一个全局池化层，输出一个2D张量，shape为(batch_size, channels)
    # "max" 代表全局最大池化，GlobalMaxPooling2D
    # https://keras.io/zh/applications/
    classes=1000)  # 类别数
