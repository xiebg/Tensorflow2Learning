

# @tf.function装饰器 将函数转换为图模式，提高性能
# 使用静态编译将函数内的代码转换为计算图
# 语句有限制（仅支持python语言的子集），且需要函数内的操作本身可以被构建为计算图，建议使用tensorflow的原生操作
# 参数最好只包括tensorflow的tensor对象和numpy数组
# 优点：性能提升（尤其较多小操作）
# 内在机制：第一次调用时：
# 1. 关闭eager模式，开启图模式，只定义计算节点，不执行计算
# 2. 使用AutoGraph将python代码转换为计算图节点，并将节点添加到计算图中（图中会自动加入tf.control_dependencies()节点保证顺利执行）
# 3. 运行计算图
# 4. 基于函数名称和参数类型生成一个哈希值，并将建立的计算图缓存到哈希表中
# 5. 下次调用相同函数时，首先检查哈希表中是否存在缓存的计算图，如果存在，则直接使用缓存的计算图，否则重新构建计算图

# 注意：当有多个函数实现不同的运算时，仅需在最后一个函数上使用@tf.function装饰器，其他函数将自动使用静态编译模式


import tensorflow as tf
import numpy as np

def read_jpg(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def normalization(input_img):
    input_img = tf.image.resize(input_img, [scal, scal])
    input_img = tf.cast(input_img, tf.float32) / 127.5 - 1
    return input_img

@tf.function
def load_image(path):
    input_img = read_jpg(path)
    input_img = normalization(input_img)
    return input_img

# GPU配置与使用加速

gpus = tf.config.experimental.list_physical_devices('GPU') # 获取GPU设备列表
cpus = tf.config.experimental.list_physical_devices('CPU') # 获取CPU设备列表
# 方法一
tf.config.experimental.set_visible_devices(gpus[0:2], 'GPU') # 设置GPU设备,设置前两块gpu可见
# 方法二
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # 设置环境变量，仅使用GPU0和GPU1

# 设置显存使用策略
# 1.设置按需增长显存
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True) # 设置GPU0的显存按需增长
# 2.限制固定显存大小
tf.config.experimental.set_virtual_device_configuration(
    gpus[0], # 设置GPU0
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]) # 设置GPU0的显存大小为1024M