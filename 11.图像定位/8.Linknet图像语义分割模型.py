import tensorflow as tf
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# 优点：速度快，效率高，准确率高，适合大型图像语义分割任务
# 缺点：需要大量标注数据，训练时间较长

imgs_train_path = glob.glob('./data/Unet/images/train/*/*.png')
labels_train_path = glob.glob('./data/Unet/gtFine/train/*/*_gtFine_labelIds.png')

imgs_val_path = glob.glob('./data/Unet/images/val/*/*.png')
labels_val_path = glob.glob('./data/Unet/gtFine/val/*/*_gtFine_labelIds.png')

train_count = len(imgs_train_path)
val_count = len(imgs_val_path)
print('训练集数量：', train_count)  # 2975
print('验证集数量：', val_count)  # 500

# 乱序训练集
index = np.random.permutation(train_count)
imgs_train_path = np.array(imgs_train_path)[index]
labels_train_path = np.array(labels_train_path)[index]

dataset_train = tf.data.Dataset.from_tensor_slices((imgs_train_path, labels_train_path))
dataset_val = tf.data.Dataset.from_tensor_slices((imgs_val_path, labels_val_path))


def read_png(path, channels=3):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=channels)
    return image


def crop_img(img, mask):
    """
    数据增强：随机裁剪,将图片和标签图叠在一起，然后随机裁剪出256x256的图片和标签图
    :param img: 训练图
    :param mask: 标签图
    :return: 裁剪后的训练图和标签图
    """
    concat_img = tf.concat([img, mask], axis=-1)  # 叠在一起
    concat_img = tf.image.resize(concat_img, [280, 280],  # 统一尺寸
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # 双线性插值
    crop_img = tf.image.random_crop(concat_img, [256, 256, 4])  # 随机裁剪
    img = crop_img[:, :, :3]  # 原图
    mask = crop_img[:, :, 3:]  # 标签图
    return img, mask


def test_crop_img(img, mask):
    """
    测试裁剪效果
    :param img:
    :param mask:
    :return:
    """
    img, mask = crop_img(read_png(img), read_png(mask))
    plt.subplot(1, 2, 1)
    plt.imshow(img.numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(mask.numpy()))  # squeeze()去掉维度为1的维度
    plt.show()


def normal_img(img, mask):
    """
    图片预处理：归一化到[-1,1]
    :param img: 训练图
    :param mask: 标签图
    :return: 归一化后的训练图和标签图
    """
    img = tf.cast(img, tf.float32) / 127.5 - 1  # 归一化到[-1,1]
    mask = tf.cast(mask, tf.int32)
    return img, mask


def load_img_train(img_path, mask_path):
    img = read_png(img_path, 3)
    mask = read_png(mask_path, 1)
    img, mask = crop_img(img, mask)  # 随机裁剪

    if tf.random.uniform(()) > 0.5:  # 随机翻转,uniform是均匀分布
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    img, mask = normal_img(img, mask)  # 归一化
    return img, mask


def load_img_test(img_path, mask_path):
    img = read_png(img_path, 3)
    mask = read_png(mask_path, 1)

    img = tf.image.resize(img, [256, 256])
    mask = tf.image.resize(mask, [256, 256])

    img, mask = normal_img(img, mask)  # 归一化
    return img, mask


BATCH_SIZE = 4
Buffer_SIZE = 300
step_per_epoch = train_count // BATCH_SIZE
val_step_per_epoch = val_count // BATCH_SIZE
auto = tf.data.experimental.AUTOTUNE
# 加载数据集
dataset_train = dataset_train.map(load_img_train, num_parallel_calls=auto)
dataset_val = dataset_val.map(load_img_test, num_parallel_calls=auto)


# 定义卷积模块 1.conv 2. bn 3. relu
class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding='same'):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# 定义反卷积模块 1.deconv 2. bn 3. relu
class DeConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=2, padding='same'):
        super(DeConvBlock, self).__init__()
        self.deconv = tf.keras.layers.Conv2DTranspose(filters=filters,
                                                      kernel_size=kernel_size,
                                                      strides=strides,
                                                      padding=padding)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, is_activation=True):
        x = self.deconv(x)
        x = self.bn(x)
        if is_activation:
            x = self.relu(x)
        return x


# 定义编码模块
class EncodeBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(EncodeBlock, self).__init__()
        self.conv1 = ConvBlock(filters=filters, strides=2)
        self.conv2 = ConvBlock(filters=filters)

        self.conv3 = ConvBlock(filters=filters)
        self.conv4 = ConvBlock(filters=filters)

        self.shortcut = ConvBlock(filters=filters, strides=2)

    def call(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        residue = self.shortcut(x)  # 1. 先对输入进行卷积,步长为2,得到残差结构
        out2 = self.conv3(out1 + residue)  # 3. 对out1+残差结构进行卷积,步长为1,得到out2
        out2 = self.conv4(out2)
        return out1 + out2  # 4. 返回out1+out2,残差结构和out2相加作为下一层的输入


# 定义解码模块
class DecodeBlock(tf.keras.layers.Layer):
    def __init__(self, filter1, filter2):
        super(DecodeBlock, self).__init__()
        self.conv1 = ConvBlock(filters=filter1, kernel_size=1)
        self.deconv = DeConvBlock(filters=filter1)
        self.conv2 = ConvBlock(filters=filter2, kernel_size=1)

    def call(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


# 定义Linknet模型
class Linknet(tf.keras.Model):
    def __init__(self):
        super(Linknet, self).__init__()  # 256x256x3
        self.input_conv = ConvBlock(filters=64, kernel_size=7, strides=2)  # 128x128x64
        self.input_pool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')  # 64x64x64
        self.encode1 = EncodeBlock(filters=64)  # 32x32x64
        self.encode2 = EncodeBlock(filters=128)  # 16x16x128
        self.encode3 = EncodeBlock(filters=256)  # 8x8x256
        self.encode4 = EncodeBlock(filters=512)  # 4x4x512

        self.decode4 = DecodeBlock(filter1=512 // 4, filter2=256)  # 8x8x256,filter2=256是为了和encode3的输出维度一致，需要相加
        self.decode3 = DecodeBlock(filter1=256 // 4, filter2=128)  # 16x16x128
        self.decode2 = DecodeBlock(filter1=128 // 4, filter2=64)  # 32x32x64
        self.decode1 = DecodeBlock(filter1=64 // 4, filter2=64)  # 64x64x64

        self.decovn_last1 = DeConvBlock(filters=32)  # 128x128x32
        self.conv_last = ConvBlock(filters=32)  # 128x128x32
        self.conv_last2 = tf.keras.layers.Conv2D(filters=34, kernel_size=2)  # 128x128x34,分34类，不需要激活函数

    def call(self, x):
        x = self.input_conv(x)
        x = self.input_pool(x)

        e1 = self.encode1(x)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)

        d4 = self.decode4(e4) + e3
        d3 = self.decode3(d4) + e2
        d2 = self.decode2(d3) + e1
        d1 = self.decode1(d2)

        out = self.decovn_last1(d1)
        out = self.conv_last(out)
        out = self.conv_last2(out, is_activation=False)
        return out


model = Linknet()
# print(model.summary())


# 定义优化器和损失函数
opt = tf.keras.optimizers.Adam()
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


# 重新定义交并比指标
class MeanIoU(tf.keras.metrics.MeanIoU):
    def __call__(self, y_true, y_pred):
        y_pred = tf.argmax(y_pred, axis=-1)  # 将预测结果转为one-hot编码,即输出最大值的位置
        return super(MeanIoU, self).__call__(y_true, y_pred)


train_loss = tf.keras.metrics.Mean(name='train_loss')  # 定义训练损失
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')  # 定义训练准确率
train_iou = MeanIoU(num_classes=34, name='train_iou')  # 定义训练交并比

test_loss = tf.keras.metrics.Mean(name='test_loss')  # 定义测试损失
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_acc')  # 定义测试准确率
test_iou = MeanIoU(num_classes=34, name='test_iou')  # 定义测试交并比


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:  # 记录梯度信息
        predictions = model(images)  # 前向传播
        loss_value = loss(labels, predictions)  # 计算损失
    grads = tape.gradient(loss_value, model.trainable_variables)  # 计算梯度
    opt.apply_gradients(zip(grads, model.trainable_variables))  # 更新参数

    train_loss(loss_value)  # 记录训练损失
    train_acc(labels, predictions)  # 记录训练准确率
    train_iou(labels, predictions)  # 记录训练交并比


@tf.function
def test_step(images, labels):  # 测试过程
    predictions = model(images)  # 前向传播
    t_loss = loss(labels, predictions)  # 计算损失

    test_loss(t_loss)  # 记录测试损失
    test_acc(labels, predictions)  # 记录测试准确率
    test_iou(labels, predictions)  # 记录测试交并比


model_path = r'D:\Project\pythonProject\models\Linknet_model.h5'


def train_model(Epochs):  # 训练模型
    for epoch in range(Epochs):  # 开始训练
        train_loss.reset_states()  # 重置训练损失
        train_acc.reset_states()  # 重置训练准确率
        train_iou.reset_states()  # 重置训练交并比

        test_loss.reset_states()  # 重置测试损失
        test_acc.reset_states()  # 重置测试准确率
        test_iou.reset_states()  # 重置测试交并比

        print("总训练步数：", step_per_epoch)
        print("总测试步数：", val_step_per_epoch)
        for step, (images, labels) in enumerate(dataset_train):  # 训练过程
            train_step(images, labels)  # 训练一步
            if step % 10 == 0:  # 每10步打印一次训练信息
                print(
                    f'Epoch {epoch + 1} 训练步数 {step + 1} 损失 {train_loss.result():.4f} 准确率 {train_acc.result():.4f} 交并比 {train_iou.result():.4f}')

        for test_images, test_labels in dataset_val:  # 测试过程
            test_step(test_images, test_labels)  # 测试一步

        print(
            f'Epoch {epoch + 1} 训练集损失 {train_loss.result():.4f} 准确率 {train_acc.result():.4f} 交并比 {train_iou.result():.4f}')
        print(
            f'Epoch {epoch + 1} 测试集损失 {test_loss.result():.4f} 准确率 {test_acc.result():.4f} 交并比 {test_iou.result():.4f}')
        print('=' * 50)




# Epochs = 5  # 训练轮数
# train_model(Epochs)  # 开始训练




# 预测模型
def predict(model, image):
    image = tf.expand_dims(image, axis=0)
    image = tf.image.resize(image, [256, 256])
    image = normal_img(image, None)[0]
    pred = model(image)
    pred = tf.argmax(pred, axis=-1)
    return pred.numpy()


def predict_image():
    # 预测测试集
    for img_path, label_path in zip(imgs_val_path, labels_val_path):
        img = read_png(img_path, 3)
        label = read_png(label_path, 1)
        pred = predict(model, img)
        plt.subplot(1, 3, 1)
        plt.imshow(img.numpy())
        plt.subplot(1, 3, 2)
        plt.imshow(np.squeeze(label.numpy()))
        plt.subplot(1, 3, 3)
        plt.imshow(np.squeeze(pred))
        plt.show()

predict_image()  # 预测测试集