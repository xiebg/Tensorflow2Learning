import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# U-Net模型前半部分是特征提取器，后半部分是语义分割器。
#
# 数据集：城市街景数据集：https://www.cityscapes-dataset.com/
# 20000粗标注图像，5000精标注图像，
# 精标中有2975个训练图像，500个验证图像，标签34种类别. Test不提供标签.
# 数据集包含两部分：Images和gtFine，前者是原始图像（2048x1024），后者是精细标注的图像。
# _gtFine_color代表彩色语义分割图，_gtFine_instance代表实例分割图，_gtFine_labelIds代表标签图.

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


img_1 = imgs_train_path[2]
label_1 = labels_train_path[2]


# 数据增强
# 1. 随机同时翻转 img = tf.image.flip_left_right()
# 2. 随机同时裁剪（叠在一起裁剪）concat_img = tf.concat([img1, img2], axis=-1)
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


# test_crop_img(img_1, label_1)

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
Buffer_SIZE = 300  # 乱序缓冲区大小，越大越乱，这里小的话前面必须足够充分乱序，否则会在同一城市乱序
step_per_epoch = train_count // BATCH_SIZE
val_step_per_epoch = val_count // BATCH_SIZE
auto = tf.data.experimental.AUTOTUNE  # 自动选择最优配置

# 加载数据集
dataset_train = dataset_train.map(load_img_train, num_parallel_calls=auto)
dataset_val = dataset_val.map(load_img_test, num_parallel_calls=auto)


def test_dataset(dataset):
    for img, mask in dataset.take(1):
        plt.subplot(1, 2, 1)
        plt.imshow((img.numpy() + 1) / 2)
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(mask.numpy()))  # squeeze()去掉维度为1的维度
        plt.show()


# test_dataset(dataset_train)

dataset_train = dataset_train.shuffle(Buffer_SIZE).batch(BATCH_SIZE)
dataset_val = dataset_val.batch(BATCH_SIZE)


# 定义unet模型
class DownSample(tf.keras.layers.Layer):
    def __init__(self, units):  # 定义下采样层
        super(DownSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(units, 3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(units, 3, padding='same')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))

    def call(self, x, is_pool=True):  # 定义下采样过程
        if is_pool:
            x = self.pool(x)
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        return x


class UpSample(tf.keras.layers.Layer):
    def __init__(self, units):  # 定义上采样层
        super(UpSample, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(units, kernel_size=3, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(units, kernel_size=3, padding='same')
        self.deConv = tf.keras.layers.Conv2DTranspose(units // 2, kernel_size=2, strides=2, padding='same')

    def call(self, x, is_deConv=True):  # 定义上采样过程
        x = self.conv1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        if is_deConv:
            x = self.deConv(x)
            x = tf.nn.relu(x)
        return x


class UnetModel(tf.keras.Model):
    def __init__(self):  # 定义模型结构
        super(UnetModel, self).__init__()
        self.down1 = DownSample(64)
        self.down2 = DownSample(128)
        self.down3 = DownSample(256)
        self.down4 = DownSample(512)
        self.down5 = DownSample(1024)

        self.up = tf.keras.layers.Conv2DTranspose(512,
                                                  kernel_size=2,
                                                  strides=2,
                                                  padding='same')
        self.up1 = UpSample(512)
        self.up2 = UpSample(256)
        self.up3 = UpSample(128)
        self.up4 = UpSample(64)
        self.last_conv = tf.keras.layers.Conv2D(34,
                                                kernel_size=1,
                                                padding='same')

    def call(self, x):  # 前向传播
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up(x5)
        x = tf.concat([x, x4], axis=-1)
        x = self.up1(x)
        x = tf.concat([x, x3], axis=-1)
        x = self.up2(x)
        x = tf.concat([x, x2], axis=-1)
        x = self.up3(x)
        x = tf.concat([x, x1], axis=-1)
        x = self.up4(x, is_deConv=False)
        x = self.last_conv(x)
        return x


model = UnetModel()

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


# 模型保存检查点的配置
model_Checkpoint_path = r'D:\Project\pythonProject\models\save_Unet_model.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_Checkpoint_path, # 保存路径
                                                 # save_weights_only=True, # 只保存模型权重
                                                 verbose=1) # 保存频率,每n轮保存一次

# 保存目录
cp_dir = r'D:\Project\pythonProject\models\Unet_model'
cp_prefix = os.path.join(cp_dir, 'ckpt') # 设置模型前缀
checkpoint = tf.train.Checkpoint( # 设置需要保存的内容
    optimizer=opt,
    model=model
)

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

        # 保存模型
        checkpoint.save(file_prefix=cp_prefix)  # 每轮保存一次模型


# Epochs = 5  # 训练轮数
# train_model(Epochs)  # 开始训练

# 加载模型
new_point_path = tf.train.latest_checkpoint(cp_dir) # 获取最新检查点的模型路径
checkpoint.restore(new_point_path)


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
