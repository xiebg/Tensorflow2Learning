import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 加载数据集
all_image_paths = glob.glob('.\\data\\2_class\\*\\*.jpg')
# print(all_image_paths[0:5])

# 乱序
import random
random.shuffle(all_image_paths)
# print(all_image_paths[0:5])

# # 定义标签
# all_image_labels = [path.split(os.path.sep)[-2] for path in all_image_paths]
# print(all_image_labels)

# 定义标签
label_to_index = {'airplane': 0, 'lake': 1}
# 标签转换为索引
index_to_label = dict((v,k) for k,v in label_to_index.items())

# img_path = all_image_paths[0]
# a = label_to_index[img_path.split("\\")[-2]]
# print(a)

# 获取所有图片的标签并转换为索引
all_image_labels = [(label_to_index[img_path.split("\\")[-2]]) for img_path in all_image_paths]
print(all_image_labels[0:5]) # [0, 1, 0, 0, 0]

# 加载图片
def load_image(img_path,channels=3):
    # img_path = all_image_paths[0]
    img_raw = tf.io.read_file(img_path) # 读取图片 二进制格式
    # 解码图片
    img_tensor = tf.image.decode_jpeg(img_raw,channels=channels) # channels=3表示RGB三通道 decode_image()是通用的
    # 调整图片大小
    img_tensor = tf.image.resize(img_tensor, [256, 256]) # 调整图片大小为256x256
    print(img_tensor.shape,img_tensor.dtype) # (256, 256, 3) <dtype: 'uint8'> uint8不适合做模型输入，需要转换为float32
    # 转换为float32
    img_tensor = tf.cast(img_tensor, tf.float32) # 将uint8转换为float32
    # 归一化
    img_tensor = img_tensor/255.0 # 归一化到0-1之间
    # print(img_tensor.numpy().max(),img_tensor.numpy().min())
    return img_tensor

def text_a_image():
    # 随机选择一张图片测试
    i = random.choice(range(len(all_image_paths)))
    img_path = all_image_paths[i]
    label = all_image_labels[i]
    img_tensor = load_image(img_path)

    plt.title(index_to_label[label])
    plt.imshow(img_tensor.numpy())
    plt.show()


# text_a_image()
# 创建数据集
img_path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths) # 创建一个数据集，包含所有图片的路径
img_ds = img_path_ds.map(load_image) # 加载图片，并将图片转换为张量
# print(img_ds) # <TensorSliceDataset shapes: (256, 256, 3), types: tf.float32>

label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(all_image_labels, tf.int64)) # 创建一个数据集，包含所有图片的标签
# print(label_ds) # <TensorSliceDataset shapes: (), types: tf.int64>
# for img, label in zip(img_ds, label_ds):
#     print(img.shape, label.numpy()) # (256, 256, 3) 0
#     break

# 合并数据集
image_label_ds = tf.data.Dataset.zip((img_ds, label_ds)) # 合并数据集
# print(image_label_ds) # <MapDataset shapes: ((256, 256, 3), ()), types: (tf.float32, tf.int64)>

# 划分训练集和测试集
train_count = int(len(all_image_paths)*0.8) # 训练集数量 , 取前80%作为训练集
test_count = len(all_image_paths) - train_count # 测试集数量
train_ds = image_label_ds.take(train_count)
test_ds = image_label_ds.skip(test_count) # 取后20%作为测试集,跳过前80%

# 数据集预处理
BATCH_SIZE = 16
train_ds = train_ds.repeat().shuffle(1000).batch(BATCH_SIZE)    # 训练集重复、打乱、分批
test_ds = test_ds.batch(BATCH_SIZE)    # 测试集分批

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(256, 256, 3), padding='same'), # 卷积层
    tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(), # 池化层
    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu',padding='same'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    tf.keras.layers.GlobalMaxPooling2D(), # 全局池化层
    tf.keras.layers.Dense(512, activation='relu'), # 全连接层
    tf.keras.layers.Dense(128, activation='relu'), # 全连接层 128个神经元
    tf.keras.layers.Dense(1, activation='sigmoid') # 输出层 大于0.5为飞机，小于0.5为湖泊 1个神经元，输出为1维概率值
])
# 定义模型 批标准化
model2 = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(256, 256, 3), padding='same'), # 卷积层
    tf.keras.layers.BatchNormalization(), # 批标准化层
    tf.keras.layers.Activation('relu'), # 激活层

    tf.keras.layers.Conv2D(64, (3,3),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(), # 池化层
    tf.keras.layers.Conv2D(128, (3,3),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Conv2D(128, (3,3),padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, (3,3), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),

    tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.GlobalMaxPooling2D(), # 全局池化层
    tf.keras.layers.Dense(512), # 全连接层
    tf.keras.layers.BatchNormalization(), # 批标准化层
    tf.keras.layers.Activation('relu'), # 激活层

    tf.keras.layers.Dense(128), # 全连接层 128个神经元
    tf.keras.layers.BatchNormalization(), # 批标准化层
    tf.keras.layers.Activation('relu'), # 激活层
    tf.keras.layers.Dense(1, activation='sigmoid') # 输出层 大于0.5为飞机，小于0.5为湖泊 1个神经元，输出为1维概率值
])


# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              # optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), # from_logits表示是否将预测值激活，默认是False
              # binary_crossentropy()是方法名，需要传入参数预测值和标签；BinaryCrossentropy是类名
              # loss='binary_crossentropy',
              metrics=['accuracy']) # accuracy可以简写为acc
# 训练模型

steps_per_epoch = train_count // BATCH_SIZE # 计算每个epoch需要的步数
validation_steps = test_count // BATCH_SIZE # 计算每个epoch需要的验证步数
history = model.fit(train_ds, epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_ds,
                    validation_steps=validation_steps
                    )

def show_train_loss_info(history,name):
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend() # 显示图例
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.savefig(f'./img/{name}.png')
    plt.show()

def show_train_accuracy_info(history,name):
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.legend() # 显示图例
    plt.savefig(f'./img/{name}.png')
    plt.show()

show_train_accuracy_info(history,'airplane_or_lake_accuracy')
# 保存模型
model.save('airplane_or_lake.h5')

# 预测
# model = tf.keras.models.load_model('airplane_or_lake.h5')
img_path = all_image_paths[45]
img_tensor = load_image(img_path)
img_tensor = tf.expand_dims(img_tensor, axis=0) # 增加一个维度
result = model.predict(img_tensor)
print(result) # [0.9999999]
out = index_to_label[(result>0.5).astype(int)][0][0] # 'airplane'(result>0.5).astype(int) # True
print(out) # airplane

def predict_a_image(img_path):
    img_tensor = load_image(img_path)
    img_tensor = tf.expand_dims(img_tensor, axis=0) # 增加一个维度
    result = model.predict(img_tensor)
    out = index_to_label[(result>0.5).astype(int)][0][0] # 'airplane'(result>0.5).astype(int) # True
    print(out)
    return out

# predict_a_image('.\\data\\2_class\\airplane_0001.jpg')