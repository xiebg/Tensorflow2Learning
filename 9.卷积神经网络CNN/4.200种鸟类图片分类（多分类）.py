import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 加载数据集
all_image_paths = glob.glob('.\\data\\birds\\*\\*.jpg')

all_image_labels = [img_path.split("\\")[-2].split(".")[1] for img_path in all_image_paths]
# 唯一的标签
unique_labels = set(all_image_labels)

# 标签到索引的映射
label_to_index = {label: index for index, label in enumerate(unique_labels)}
# 索引到标签的映射
index_to_label = {index: label for label, index in label_to_index.items()}
# 所有标签的索引
all_image_labels = [label_to_index[label] for label in all_image_labels]

np.random.seed(42)  # 设置随机种子
random_indexs = np.random.permutation(len(all_image_paths))  # 随机打乱数据集

all_image_paths = np.array(all_image_paths)[random_indexs]
all_image_labels = np.array(all_image_labels)[random_indexs]
# 定义数据集大小
train_size = int(len(all_image_paths) * 0.8)
test_size = len(all_image_paths) - train_size
# 划分训练集和测试集
train_image_paths = all_image_paths[:train_size]
train_image_labels = all_image_labels[:train_size]
test_image_paths = all_image_paths[train_size:]
test_image_labels = all_image_labels[train_size:]
# 创建数据集对象
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))

# 加载图片
def load_image(img_path,label,channels=3):
    # img_path = all_image_paths[0]
    img_raw = tf.io.read_file(img_path) # 读取图片 二进制格式
    # 解码图片
    img_tensor = tf.image.decode_jpeg(img_raw,channels=channels) # channels=3表示RGB三通道 decode_image()是通用的
    # 调整图片大小
    img_tensor = tf.image.resize(img_tensor, [256, 256]) # 调整图片大小为256x256
    # print(img_tensor.shape,img_tensor.dtype) # (256, 256, 3) <dtype: 'uint8'> uint8不适合做模型输入，需要转换为float32
    # 转换为float32
    img_tensor = tf.cast(img_tensor, tf.float32) # 将uint8转换为float32
    # 归一化
    img_tensor = img_tensor/255.0 # 归一化到0-1之间
    # print(img_tensor.numpy().max(),img_tensor.numpy().min())
    return img_tensor,label

# 加载图片并裁剪数据集
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 自动调整数据集的并行处理
train_dataset = train_dataset.map(load_image, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(load_image, num_parallel_calls=AUTOTUNE)
# 数据集预处理
BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(200).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# 定义模型 批标准化
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), input_shape=(256, 256, 3), padding='same', activation='relu'), # 卷积层
    tf.keras.layers.BatchNormalization(), # 批标准化层
    tf.keras.layers.Conv2D(64, (3,3),padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(), # 批标准化层
    tf.keras.layers.MaxPooling2D(), # 池化层

    tf.keras.layers.Conv2D(128, (3,3),padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3,3),padding='same',activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.MaxPooling2D(),

    # tf.keras.layers.Conv2D(256, (3,3), padding='same',activation='relu'),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Conv2D(256, (3,3), activation='relu',padding='same'),
    # tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalMaxPooling2D(), # 全局池化层,将特征图转换为1维向量，比Flatten层更高效

    # tf.keras.layers.Dense(512, activation='relu'), # 全连接层
    # tf.keras.layers.BatchNormalization(), # 批标准化层
    tf.keras.layers.Dense(128,activation='relu'), # 全连接层 128个神经元
    tf.keras.layers.BatchNormalization(), # 批标准化层
    # tf.keras.layers.Dense(200, activation='softmax') # 输出层
    tf.keras.layers.Dense(200) # 可以不用激活函数，这种情况下输出值就是logits，可以用于后续计算loss，输出为长度200的张量，多少个类就输出长度多少的张量
    # 这200个神经元对应着200种鸟类，softmax激活函数用于分类，输出概率分布，哪个类概率最大就属于哪个类
    # 将激活函数放在损失函数中计算，更加稳定和高效
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # from_logits表示之前没有激活函数，这里需要激活函数
              metrics=['accuracy'])

# 定义训练参数
train_count = len(train_image_paths)
test_count = len(test_image_paths)

steps_per_epoch = train_count // BATCH_SIZE
validation_steps = test_count // BATCH_SIZE
# 训练模型
history = model.fit(train_dataset,
                    epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    validation_data=test_dataset,
                    validation_steps=validation_steps)

def show_train_accuracy_info(history,name):
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.plot(history.epoch, history.history['accuracy'], label='accuracy')
    plt.plot(history.epoch, history.history['val_accuracy'], label='val_accuracy')
    plt.legend() # 显示图例
    plt.savefig(f'./img/{name}.png')
    plt.show()

show_train_accuracy_info(history,'bird_accuracy')
# # 保存模型
# model.save('bird_classification.h5')

def load_and_preprocess_image(img_path,channels=3):
    # 加载图片
    img_raw = tf.io.read_file(img_path) # 读取图片 二进制格式
    # 解码图片
    img_tensor = tf.image.decode_jpeg(img_raw,channels=channels) # channels=3表示RGB三通道 decode_image()是通用的
    # 调整图片大小
    img_tensor = tf.image.resize(img_tensor, [256, 256]) # 调整图片大小为256x256
    # 转换为float32
    img_tensor = tf.cast(img_tensor, tf.float32) # 将uint8转换为float32
    # 归一化
    img_tensor = img_tensor/255.0 # 归一化到0-1之间
    return img_tensor

test_image_paths = ".\\data\\birds\\026.Bronzed_Cowbird\\Bronzed_Cowbird_0019_796242.jpg"
test_tensor = load_and_preprocess_image(test_image_paths)
test_tensor = tf.expand_dims(test_tensor,axis=0) # 增加一个维度,表示batch维度
perdict_result = model.predict(test_tensor) # 返回的预测结果是一个长度为200的张量（np.array）
print(perdict_result)
print(index_to_label[np.argmax(perdict_result)])

