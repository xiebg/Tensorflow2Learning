import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# 对每个点进行分类，分为背景、边缘、前景等
# 不是实例分割，不能区分同一类物体

# 输入图片（任意尺寸，任意通道），
# 输出分割图（与输入图片相同尺寸），其中每个像素包含了其所属的类别wight * height * 1，通道数为n(目标类别数)+1（背景）
# 算法 1：FCN（Fully Convolutional Network）全卷积网络（开始），全部卷积层，全部反卷积层，不用全连接层
# 算法 2：UNet（U-Net） 2015,当前最广泛使用的分割算法
# 算法 3：SegNet（SegNet）

# 下采样：
# 1.全局池化 2.平均池化 3.最大池化 4.卷积步长
# 上采样：将分割图上采样到与输入图片相同尺寸
# 1.插值法（插入前一个像素与后一个像素的均值） 2.反池化（反最大池化：其他像素插入0；反平均池化：其他像素插入均值）
# 3.反卷积（转置卷积），最常用：tf.keras.layers.Conv2DTranspose
# FCN缺点：1.结果不够精细，只能分割出大概的轮廓 2.没有考虑像素之间的关系，缺乏空间一致性

# label_paths = os.listdir(r".\data\oxford\annotations\trimaps")
label_paths = glob.glob(r".\data\oxford\annotations\trimaps\*.png")


def test_label_img():
    index = 10
    # path = r".\data\oxford\annotations\trimaps\\" + label_paths[index]
    path = label_paths[index]
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img)
    img = tf.squeeze(img)  # 去掉单维度,[358,500,1]变为[358,500]
    plt.imshow(img)
    plt.show()
    print(np.unique(img.numpy()))  # 1是主体，2是背景，3是边缘

    path2 = r".\data\oxford\images\\" + label_paths[index].replace(".png", ".jpg")
    img2 = tf.io.read_file(path2)
    img2 = tf.image.decode_jpeg(img2)
    plt.imshow(img2)
    plt.show()


# test_label_img()
image_paths = glob.glob(r".\data\oxford\images\*.jpg")
print(len(image_paths))  # 7390
print(len(label_paths))  # 7390
# 数量相同，并一一对应

# 乱序
np.random.seed(2024)
index = np.random.permutation(len(image_paths))
image_paths = np.array(image_paths)[index]
label_paths = np.array(label_paths)[index]

# 划分训练集和测试集
dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
test_count = int(len(image_paths) * 0.2)
train_count = len(image_paths) - test_count
train_dataset = dataset.take(train_count)  # 训练集，前80%
test_dataset = dataset.skip(train_count)


def read_jpg(path):
    img = tf.io.read_file(path)  # 读取图片
    img = tf.image.decode_jpeg(img, channels=3)  # 解码
    return img  # 返回图片


def read_png(path):
    img = tf.io.read_file(path)  # 读取图片
    img = tf.image.decode_png(img, channels=1)  # 解码
    return img  # 返回图片


def normal_img(input_img, label_img):
    """
    归一化图片
    :param input_img: 训练图片
    :param label_img: 目标图片
    :return: 归一化后的图片
    """
    input_img = tf.cast(input_img, tf.float32) / 127.5 - 1  # 归一化到[-1,1]
    label_img = tf.cast(label_img, tf.int32)  # 转为int32
    label_img -= 1  # 将[1,2,3]转为[0,1,2]
    return input_img, label_img


def load_images(image_path, label_path):  # 加载图片
    input_img = read_jpg(image_path)  # 读取图片
    label_img = read_png(label_path)  # 读取标签
    input_img = tf.image.resize(input_img, [224, 224])  # 调整图片大小
    label_img = tf.image.resize(label_img, [224, 224])  # 调整标签大小
    return normal_img(input_img, label_img)  # 归一化图片


train_dataset = train_dataset.map(load_images,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 加载图片
test_dataset = test_dataset.map(load_images,
                                num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 加载图片

BATCH_SIZE = 64
train_dataset = train_dataset.repeat().shuffle(100).batch(BATCH_SIZE)  # 重复,打乱,分批
test_dataset = test_dataset.batch(BATCH_SIZE)  # 分批


def show_a_dataset(dataset):
    """
    测试数据集，显示数据集的第一张图片和标签
    :param dataset:
    :return:
    """
    for img, label in test_dataset.take(1):  # 取出一批图片
        plt.subplot(1, 2, 1)
        # plt.imshow(img[0].numpy().astype(np.uint8))  # 显示图片
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))  # 显示图片,取出第一张图片，并转为图片格式
        plt.subplot(1, 2, 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(label[0]))  # 显示标签
        plt.show()


# show_a_dataset(train_dataset)

conv_base = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3))
conv_base.trainable = False  # 冻结卷积层
conv_base.summary()

# (None, 7, 7, 512) 上采样 为 (None, 14, 14, 512)，再与前面一层相加，得到 (None, 14, 14, 512)。
# 然后再上采样为 (None, 28, 28, 512)，再与前面一层相加，得到 (None, 28, 28, 512)。后面统一为上采样为
# 然后再上采样为 (None, 56, 56, 512)，再上采样为 (None, 112, 112, 512)，最后上采样为 (None, 224, 224, 512)。
#
# conv_base.layers # 查看所有层，输出列表，可以通过切片获取
# conv_base.get_layer(name) # 通过名字获取层
# conv_base.input # 输入层
# conv_base.output # 输出层
# 获取某一层的权重和偏置
# conv_base.get_layer("block5_conv3").weights # 输出权重和偏置
# conv_base.get_layer("block5_conv3").kernel # 输出权重


# 创建一个子模型，获取block5_conv3层的输出
sub_model = tf.keras.models.Model(inputs=conv_base.input,
                                  outputs=conv_base.get_layer("block5_conv3").output
                                  )
# 最后一层输出可以通过model.predict()获取 或者通过model.layers[-1].output获取


# 创建多输出模型，输入到block5_conv3层，输出到block4_conv3层，block3_conv3层，block2_conv2层，block1_conv2层
layer_names = ["block5_conv3",
               "block4_conv3",
               "block3_conv3",
               "block2_conv2",
               # "block1_conv2"
               "block5_pool"
               ]
layer_outputs = [conv_base.get_layer(name).output for name in layer_names]
multi_output_model = tf.keras.models.Model(inputs=conv_base.input,
                                           outputs=layer_outputs)
multi_output_model.trainable = False  # 冻结卷积层

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
out_list = multi_output_model(inputs)
output = out_list[-1]

x1 = tf.keras.layers.Conv2DTranspose(
    filters=512,  # 输出通道数,卷积核数
    kernel_size=3,  # 卷积核大小
    strides=2,  # 步长，可以放大
    padding="same",  # 填充方式，same表示边界填充
    activation="relu"  # 激活函数
)(output)
x1 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x1)  # 进一步提取特征
x2 = tf.add(x1, out_list[0])  # (None, 14, 14, 512)

x2 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=3, strides=2, padding="same", activation="relu")(
    x2)  # 上采样
x2 = tf.keras.layers.Conv2D(512, 3, activation="relu", padding="same")(x2)  # 进一步提取特征
x3 = tf.add(x2, out_list[1])  # (None, 28, 28, 512)

x3 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding="same", activation="relu")(
    x3)  # 上采样
x3 = tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same")(x3)  # 进一步提取特征
x4 = tf.add(x3, out_list[2])  # (None, 56, 56, 256)

x4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding="same", activation="relu")(
    x4)  # 上采样
x4 = tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same")(x4)  # 进一步提取特征
x5 = tf.add(x4, out_list[3])  # (None, 112, 112, 128)
# 最后一层输出
prediction = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding="same",
                                             activation="softmax")(x5)  # 上采样
# (None, 224, 224, 3)

model = tf.keras.models.Model(inputs=inputs, outputs=prediction)

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


def plot_history(acc, val_acc, loss, val_loss):
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


save_path = r"D:\Project\pythonProject\models\semantic_segmentation.h5"


def train():
    # 训练模型
    history = model.fit(train_dataset,
                        epochs=10,
                        steps_per_epoch=int(train_count / BATCH_SIZE),
                        validation_data=test_dataset,
                        validation_steps=int(test_count / BATCH_SIZE))

    # 保存模型
    model.save(save_path)

    # 显示训练过程
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plot_history(acc, val_acc, loss, val_loss)


train()
# 加载模型
model = tf.keras.models.load_model(save_path)

# 预测
num = 3  # 取3张图片
for image, mask in test_dataset.take(1):  # 取一个batch
    pred_mask = model.predict(image)  # 预测mask (None, 224, 224, 3)
    pred_mask = tf.argmax(pred_mask, axis=-1)  # 取最大值索引,对最后一维进行操作,(None, 224, 224),即对三个通道的三个值取最大值索引
    pred_mask = pred_mask[..., tf.newaxis]  # 增加通道维度，...表示前面维度不变,(224, 224, 1)

    plt.figure(figsize=(10, 10))
    for i in range(num):  # 取num张图片
        # 原图
        plt.subplot(num, 3, i * num + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image[i]))
        # 标签图
        plt.subplot(num, 3, i * num + 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(mask[i]))
        # 预测图
        plt.subplot(num, 3, i * num + 3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[i]))
        plt.axis("off")
    plt.show()
