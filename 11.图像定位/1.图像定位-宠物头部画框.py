# 常见的图像处理任务：
# 1. 图像分类
# 2. 图像分类+定位（画框） 模型：多输出网络
# 3. 语义分割（区分不同类别的像素），模型：上采样Fcn，Unet,Linknet
# 4. 目标检测（检测出图像中有哪些目标，并标注出其相对位置）
# 5. 实例分割 （检测出图像中有哪些目标，并标注出其相对位置，以及目标的形状）（相对于目标检测，实例分割可以更精确地分割出目标的形状）


# 图像分类+定位（画框）
# 图像定位需要输出（x,y,w,h）四个参数，分别表示目标的中心点坐标x,y，目标的宽w和高h。
# 网络架构：通过卷积积输出两个特征图，一个是分类特征图（softmax，分类值），一个是定位特征图(L2 loss,连续值)。

# 数据集 OxFord-IIIT Pet Dataset
# 37 种宠物, 每种宠物有200张图片，包含宠物分类、头部轮廓标注和语义分割信息。
# 图像大小：224x224x3。


import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import glob

import tensorflow as tf
from matplotlib.patches import Rectangle  # 画框


def test_show_img(name):
    """
    读取xml文件，获取图像的宽高，以及目标框的坐标，并在图像上画框
    :param name: 图像名称
    :return: None
    """
    xml_path = f"./data/oxford/annotations/xmls/{name}.xml"
    xml = open(xml_path, 'rb').read()
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    hight = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//object/bndbox/xmin/text()')[0])
    ymin = int(sel.xpath('//object/bndbox/ymin/text()')[0])
    xmax = int(sel.xpath('//object/bndbox/xmax/text()')[0])
    ymax = int(sel.xpath('//object/bndbox/ymax/text()')[0])
    print(width, hight, xmin, ymin, xmax, ymax)

    img_path = f"./data/oxford/images/{name}.jpg"
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    plt.imshow(img)
    # 画框
    rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                     edgecolor='red')  # 画框 参数：(左上角x,y),宽,高,是否填充，边框颜色,
    ax = plt.gca()  # 获取当前图像的坐标轴
    ax.axes.add_patch(rect)  # 在坐标轴上添加框

    plt.show()

# test_show_img("Abyssinian_11")

def test_show_img_resize(name):
    """
    读取xml文件，获取图像的宽高，以及目标框的坐标，并在图像上画框
    缩放图片，并在缩放后的图片上画框
    :param name: 图像名称
    :return: None
    """
    xml_path = f"./data/oxford/annotations/xmls/{name}.xml"
    xml = open(xml_path, 'rb').read()
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    hight = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//object/bndbox/xmin/text()')[0])
    ymin = int(sel.xpath('//object/bndbox/ymin/text()')[0])
    xmax = int(sel.xpath('//object/bndbox/xmax/text()')[0])
    ymax = int(sel.xpath('//object/bndbox/ymax/text()')[0])
    print(width, hight, xmin, ymin, xmax, ymax)
    img_path = f"./data/oxford/images/{name}.jpg"
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224]) / 255.0  # 重置大小为224x224，并归一化
    plt.imshow(img)
    x_min = (xmin / width) * 224
    y_min = (ymin / hight) * 224
    x_max = (xmax / width) * 224
    y_max = (ymax / hight) * 224
    # 画框
    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False,
                     edgecolor='red')  # 画框 参数：(左上角x,y),宽,高,是否填充，边框颜色,
    ax = plt.gca()  # 获取当前图像的坐标轴
    ax.axes.add_patch(rect)  # 在坐标轴上添加框
    plt.show()

# test_show_img_resize("Abyssinian_11")
# 数据预处理，先处理标注
# 创建输入管道
image_paths = glob.glob("./data/oxford/images/*.jpg")
xml_paths = glob.glob("./data/oxford/annotations/xmls/*.xml")

# 所有有标注的图像名称
names = [path.split("\\")[-1].split(".")[0] for path in xml_paths]
# 所有有标注的图像路径作为训练集
# image_paths_train = [f"./data/oxford/images/{name}.jpg" for name in names]
image_paths_train = [img_path for img_path in image_paths if (img_path.split("\\")[-1].split(".")[0]) in names]

test_image_paths = [img_path for img_path in image_paths if (img_path.split("\\")[-1].split(".")[0]) not in names]
# 排序,按照文件名排序,使其对应
image_paths_train.sort(key=lambda x: x.split("\\")[-1].split(".")[0])
xml_paths.sort(key=lambda x: x.split("\\")[-1].split(".")[0])

def parse_xml(xml_path):
    """
    解析xml文件，获取图像的宽高，以及目标框的坐标,以比值形式(归一化)返回
    :param xml_path: xml文件路径
    :return: 【xmin_norm, ymin_norm, xmax_norm, ymax_norm, width, hight】
    """
    xml = open(xml_path, 'rb').read()
    sel = etree.HTML(xml)
    width = int(sel.xpath('//size/width/text()')[0])
    hight = int(sel.xpath('//size/height/text()')[0])
    xmin = int(sel.xpath('//object/bndbox/xmin/text()')[0])
    ymin = int(sel.xpath('//object/bndbox/ymin/text()')[0])
    xmax = int(sel.xpath('//object/bndbox/xmax/text()')[0])
    ymax = int(sel.xpath('//object/bndbox/ymax/text()')[0])
    # return xmin, ymin, xmax, ymax, width, hight
    return [xmin / width, ymin / hight, xmax / width, ymax / hight, width, hight]

all_labels = [parse_xml(xml_path) for xml_path in xml_paths]
out1,out2,out3,out4,out5,out6 = list(zip(*all_labels)) # 解包,将列表中的元素x_min,y_min等值拆开，分别赋值给out1,uot2等,zip(*all_labels)表示将all_labels中的元素打包成元组，然后解包
# out1 = x_min_norm, out2 = y_min_norm, out3 = x_max_norm, out4 = y_max_norm, out5 = width, out6 = hight
out1 = np.array(out1)
out2 = np.array(out2)
out3 = np.array(out3)
out4 = np.array(out4)
labels_dataset = tf.data.Dataset.from_tensor_slices((out1, out2, out3, out4))

def load_image(image_path):
    img = tf.io.read_file(image_path) # 读取图片
    img = tf.image.decode_jpeg(img, channels=3) # 解码图片
    img = tf.image.resize(img, [224, 224])
    img = img / 127.5 - 1 # 归一化,将像素值缩放到[-1,1]之间
    return img

image_dataset_path = tf.data.Dataset.from_tensor_slices(image_paths_train)
image_dataset_train = image_dataset_path.map(load_image)
dataset_train = tf.data.Dataset.zip((image_dataset_train, labels_dataset))
dataset_train = dataset_train.repeat().shuffle(buffer_size=len(image_paths_train)).batch(32)

image_dataset_path_test = tf.data.Dataset.from_tensor_slices(test_image_paths)
image_dataset_test = image_dataset_path_test.map(load_image)
dataset_test = image_dataset_test.batch(32)

def test_a_dataset(dataset_train):
    """
    测试数据集创建是否正确，显示第一张图片和标注的框
    :param dataset_train:
    :return:
    """
    for img, label in dataset_train.take(1):
        # img, label 是一个batch的图片和标签
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0])) # tf.keras.preprocessing.image.array_to_img将图片转为matplotlib可显示的格式
        out1, out2, out3, out4 = label
        x_min, y_min, x_max, y_max = out1[0].numpy() * 224, out2[0].numpy() * 224, out3[0].numpy() * 224, out4[0].numpy() * 224
        print(x_min, y_min, x_max, y_max)
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red')
        ax = plt.gca()  # 获取当前图像的坐标轴
        ax.axes.add_patch(rect)  # 在坐标轴上添加框
        plt.show()

# test_a_dataset(dataset_train)

# 构建图像定位网络模型
xception_model = tf.keras.applications.Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3))
# xception_model.trainable = False  # 冻结权重,这里我们只使用它的结构

inputs = tf.keras.layers.Input(shape=(224, 224, 3))
x = xception_model(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x)  # 全局平均池化 GlobalAveragePooling2D = Globalavgpooling2D
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)

out1 = tf.keras.layers.Dense(1, name='out1')(x)
out2 = tf.keras.layers.Dense(1, name='out2')(x)
out3 = tf.keras.layers.Dense(1, name='out3')(x)
out4 = tf.keras.layers.Dense(1, name='out4')(x)

predictions = [out1, out2, out3, out4]
model = tf.keras.Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss= "mean_squared_error", # "mse" 是均方误差,简单的线性回归模型，多输出模型，每个输出对应一个损失函数
              # metrics={'out1': tf.keras.metrics.RootMeanSquaredError(), # 多输出模型，每个输出对应一个评估指标
              #          'out2': tf.keras.metrics.RootMeanSquaredError(),
              #          'out3': tf.keras.metrics.RootMeanSquaredError(),
              #          'out4': tf.keras.metrics.RootMeanSquaredError()})
              metrics=["mae"]) # 平均绝对误差 mae = mean_absolute_error 评估指标
def train(model_path):
    # 训练模型
    EPOCHS = 10
    steps_per_epoch = len(image_paths_train) // 32
    history = model.fit(dataset_train,
                        epochs=EPOCHS,
                        steps_per_epoch=steps_per_epoch)
    # 保存模型

    model.save(model_path)
model_path = r'D:\Project\pythonProject\models\framing_model.h5'

# train(model_path)
# 加载模型
new_model = tf.keras.models.load_model(model_path)

# 测试模型
plt.figure(figsize=(8, 24))  # 画布大小
for img in dataset_test.take(1):# 取出一个batch的图片和标签
    out1, out2, out3, out4 = new_model.predict(img) # out1, out2, out3, out4 是四个batch的预测值的输出值
    for i in range(3):
        plt.subplot(3,1,i+1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[i])) # 显示图片
        x_min, y_min, x_max, y_max = out1[i] * 224, out2[i] * 224, out3[i] * 224, out4[i] * 224
        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, edgecolor='red')
        ax = plt.gca()  # 获取当前图像的坐标轴
        ax.axes.add_patch(rect)  # 在坐标轴上添加框
        plt.axis('off')  # 关闭坐标轴
    plt.show()

# 缺点
# 1.回归位置不精确
# 2.泛化能力差
# 3.目前算法只能预测一个目标
# 优化
# 1.先大后小，先预测整张图片的关键点，然后在预测出的周边进行二次预测，提高预测精度
# 2.滑动窗口预测：用一个小窗口在图片上滑动，每次做两个预测（1）是否有关键点（2）关键点的位置
# 3.针对不定个数的预测问题：可以先检测多个对象，然后对每个对象进行回归预测位置
# 4.尝试使用全卷积网络，去掉全连接层，变回归为分类问题
# 5.探索其他网络

# 图像定位的评价
# 使用 IOU（Intersection over Union）交并比 评价定位的准确度
# IOU = 预测框与真实框的交集 / 预测框与真实框的并集
# 结果越接近1，说明定位的准确度越高

# 应用
# 人体姿态估计、人脸识别、目标跟踪、视频监控、视频分析、图像检索、图像分类、图像分割、图像修复、图像超分辨率、图像风格迁移、图像生成、图像压缩、图像增强、图像检索、图像搜索、图像检索、图像检索、