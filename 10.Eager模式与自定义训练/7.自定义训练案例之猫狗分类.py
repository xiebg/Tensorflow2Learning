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

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3),activation='relu',input_shape=(256,256,3)),
    tf.keras.layers.MaxPooling2D(2,2), # 2x2的池化
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1)
])
# 优化：增加深度,增加宽度
model1 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3),input_shape=(256,256,3)),
    tf.keras.layers.BatchNormalization(), # 批标准化
    tf.keras.layers.Activation('relu'), # 激活函数
    tf.keras.layers.MaxPooling2D(), # 2x2的池化
    tf.keras.layers.Conv2D(128, (3,3)),
    tf.keras.layers.BatchNormalization(),  # 批标准化
    tf.keras.layers.Activation('relu'),  # 激活函数
    tf.keras.layers.MaxPooling2D(),  # 2x2的池化
    tf.keras.layers.Conv2D(256, (3,3)),
    tf.keras.layers.BatchNormalization(),  # 批标准化
    tf.keras.layers.Activation('relu'),  # 激活函数
    tf.keras.layers.MaxPooling2D(),  # 2x2的池化
    tf.keras.layers.Conv2D(512, (3, 3)),
    tf.keras.layers.BatchNormalization(),  # 批标准化
    tf.keras.layers.Activation('relu'),  # 激活函数
    tf.keras.layers.MaxPooling2D(),  # 2x2的池化
    tf.keras.layers.GlobalAveragePooling2D(), # 全局平均池化,将特征图转换为1维张量
    tf.keras.layers.Dense(512),
    tf.keras.layers.BatchNormalization(),  # 批标准化
    tf.keras.layers.Activation('relu'),  # 激活函数
    tf.keras.layers.Dropout(0.5),  # 随机失活
    tf.keras.layers.Dense(256),
    tf.keras.layers.BatchNormalization(),  # 批标准化
    tf.keras.layers.Activation('relu'),  # 激活函数
    tf.keras.layers.Dense(1)
])
# 出现过拟合
# 1： 增加Dropout层
# 2： 增加数据
# 3： 图片增强（图片翻转、裁剪、旋转，曝光度，亮度，对比度等） load_train_image_aug(img_path,label,channels=3)，只需要用在训练集
# vgg16: 2个64，2个128，3个256，3个512，3个512,1个GlobalAveragePooling2D，1个4096，1个4096，1个1000，1个输出层softmax
def text():
    (imgs, labels) = next(iter(train_dataset))
    print(imgs.shape, labels.shape)
    pred = model(imgs)
    y_ = np.array([p[0].numpy() for p in tf.cast(pred>0, tf.int32)])
    y = np.array([l[0].numpy() for l in labels])
    print(y_)
    print(y)

# text()
# 自定义训练
LS = tf.keras.losses.BinaryCrossentropy() # 交叉熵损失函数,返回对象，可以通过对象调用
# print(LS([1.0,0.0,1.0,0.0],[1.0,1.0,1.0,1.0])) # tf.Tensor(7.6666193, shape=(), dtype=float32)
# print(LS([[1.0],[0.0],[1.0],[0.0]],[[1.0],[1.0],[1.0],[1.0]])) # 当前数据形状

ls = tf.keras.losses.binary_crossentropy([1.0,0.0,1.0,0.0],[1.0,1.0,1.0,1.0]) # 直接调用
# print(ls) # tf.Tensor(7.6666193, shape=(), dtype=float32)

optimizer = tf.keras.optimizers.Adam() # 优化器

epoch_loss_avg = tf.keras.metrics.Mean("train_loss") # 记录每轮的损失
train_accuracy = tf.keras.metrics.Accuracy("train_accuracy") # 记录训练集准确率

def train_step(model, images, labels): # 自定义训练步骤
    with tf.GradientTape() as tape: # 记录梯度信息
        predictions = model(images) # 预测
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, predictions) # 计算损失
    gradients = tape.gradient(loss, model.trainable_variables) # 计算损失值对模型参数的梯度
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # 更新模型参数
    epoch_loss_avg(loss) # 更新损失
    train_accuracy(labels, tf.cast(predictions>0, tf.int32)) # 更新准确率

epoch_loss_avg_test = tf.keras.metrics.Mean("test_loss") # 记录每轮的损失
test_accuracy = tf.keras.metrics.Accuracy("test_accuracy") # 记录测试集准确率

def test_step(model, images, labels): # 自定义测试步骤
    predictions = model(images,training=False) # 预测
    t_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(labels, predictions) # 计算损失
    epoch_loss_avg_test(t_loss) # 更新损失
    test_accuracy(labels, tf.cast(predictions>0, tf.int32)) # 更新准确率

train_loss_results = [] # 记录每轮的损失
train_accuracy_results = [] # 记录每轮的准确率
test_loss_results = [] # 记录每轮的损失
test_accuracy_results = [] # 记录每轮的准确率
num_epochs = 20
for epoch in range(num_epochs):
    for (images, labels) in train_dataset: # 遍历训练集
        train_step(model, images, labels) # 训练
        print('.', end='')
    print()
    train_loss_results.append(epoch_loss_avg.result()) # 记录损失
    train_accuracy_results.append(train_accuracy.result()) # 记录准确率
    print('Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}'.format(
        epoch+1,
        train_loss_results[-1],
        train_accuracy_results[-1])
    )  # 打印结果
    epoch_loss_avg.reset_states() # 重置损失
    train_accuracy.reset_states() # 重置准确率

    # 测试
    for (images, labels) in test_dataset: # 遍历测试集
        test_step(model, images, labels) # 测试
    test_loss_results.append(epoch_loss_avg_test.result()) # 记录损失
    test_accuracy_results.append(test_accuracy.result()) # 记录准确率
    print('Test Loss: {:.3f}, Test Accuracy: {:.3%}'.format(
        test_loss_results[-1],
        test_accuracy_results[-1])
    )  # 打印结果
    epoch_loss_avg_test.reset_states() # 重置损失
    test_accuracy.reset_states() # 重置准确率
