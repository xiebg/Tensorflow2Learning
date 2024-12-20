import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib

data_dir = pathlib.Path('data/dataset')
data_root = pathlib.Path(data_dir)
# for file in data_root.iterdir():
#     print(file)

all_image_paths = list(data_root.glob('*/*'))


# 打乱数据集
import random
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir()) # 获取标签名称 sorted() 函数用于对列表进行排序
# print(label_names) # ['black_jeans', 'black_shoes', 'blue_dress', 'blue_jeans', 'blue_shirt', 'red_dress', 'red_shirt']

color_label_names = set(name.split('_')[0] for name in label_names)
# print(color_label_names) # {'black', 'blue','red'}
item_label_names = set(name.split('_')[1] for name in label_names)
# print(item_label_names) # {'shirt', 'jeans', 'dress', 'shoes'}

# 会出现编码错误，两次运行结果可能不同
# color_label_to_index = {name: index for index, name in enumerate(color_label_names)}
# color_index_to_label = {index: name for index, name in enumerate(color_label_names)}
# # print(color_label_to_index) # {'black': 0, 'blue': 1,'red': 2}
# item_label_to_index = {name: index for index, name in enumerate(item_label_names)}
# item_index_to_label = {index: name for index, name in enumerate(item_label_names)}
# # print(item_label_to_index) # {'jeans': 0, 'shirt': 1, 'dress': 2, 'shoes': 3}

color_label_to_index = {'black': 0, 'blue': 1,'red': 2}
color_index_to_label = {0: 'black', 1: 'blue', 2:'red'}
item_label_to_index = {'jeans': 0,'shirt': 1, 'dress': 2,'shoes': 3}
item_index_to_label = {0: 'jeans', 1:'shirt', 2: 'dress', 3:'shoes'}

all_image_labels = [pathlib.Path(path).parent.name for path in all_image_paths]
all_color_labels = [color_label_to_index[name.split('_')[0]] for name in all_image_labels] # 获取颜色标签
all_item_labels = [item_label_to_index[name.split('_')[1]] for name in all_image_labels] # 获取物品标签

def random_show_image_display():
    """
    随机显示三张图片,display版本
    :return:
    """
    import IPython.display as display
    for i in range(3):
        image_index = random.choice(range(len(all_image_paths)))
        display.display(display.Image(all_image_paths[image_index], width=200, height=200)) # 随机显示一张图片
        print(all_image_paths[image_index]) # 打印图片路径
        print()

def random_show_image():
    """
    随机显示三张图片
    :return:
    """
    for i in range(3):
        image_index = random.choice(range(len(all_image_paths)))
        plt.subplot(1,3,i+1)
        plt.imshow(plt.imread(all_image_paths[image_index]))
        plt.title(all_image_labels[image_index])
        plt.axis('off')
    plt.show()



def load_image(img_path,channels=3):
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=channels)
    img_tensor = tf.image.resize(img_tensor, [224, 224])
    img_tensor = tf.cast(img_tensor, tf.float32)
    img_tensor = img_tensor/255.0 # 归一化到[0,1]
    img_tensor = img_tensor*2.0-1.0 # 归一化到[-1,1]
    return img_tensor
def test_show_image(): # 测试显示图片
    image_path = all_image_paths[0]
    label = all_image_labels[0]
    plt.imshow((load_image(image_path) + 1.0)/2.0)
    plt.grid(False)
    plt.xlabel(label)
    plt.show()
def show_train_accuracy_info(history,name):
    import os
    if not os.path.exists('./img'):
        os.mkdir('./img')
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['color_accuracy'], label='color_accuracy')
    plt.plot(history.epoch, history.history['val_color_accuracy'], label='val_color_accuracy')
    plt.title('color accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['item_accuracy'], label='item_accuracy')
    plt.plot(history.epoch, history.history['val_item_accuracy'], label='val_item_accuracy')
    plt.title('item accuracy')
    plt.legend() # 显示图例
    plt.savefig(f'./img/{name}.png')
    plt.show()

# 创建数据集
path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
AUTOTUNE = tf.data.experimental.AUTOTUNE
image_ds = path_ds.map(load_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices((all_color_labels,all_item_labels)) # 创建标签数据集,多输出模型
# for ele in label_ds.take(3):
#     print(ele[0].numpy(), ele[1].numpy())
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds)) # 合并数据集

image_count = len(all_image_paths)
test_count = int(image_count*0.2)
train_count = image_count - test_count

train_ds = image_label_ds.take(train_count) # 训练集
test_ds = image_label_ds.skip(train_count) # 测试集

BATCH_SIZE = 32
train_ds = train_ds.shuffle(train_count).repeat().batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE) # 创建训练集
test_ds = test_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE) # 创建测试集

# 定义模型
# 所有数据通过MobileNetV2网络处理,然后分别通过两个全连接层输出输出颜色和物品标签
inputs = tf.keras.layers.Input(shape=(224,224,3))
mobile_net = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    # weights='imagenet'
)
x = mobile_net(inputs)
x = tf.keras.layers.GlobalAveragePooling2D()(x) # 全局平均池化

x1 = tf.keras.layers.Dense(1024, activation='relu')(x) # 全连接层1
out_color = tf.keras.layers.Dense(len(color_label_names),
                                  activation='softmax',
                                  name='color')(x1) # 输出颜色标签

x2 = tf.keras.layers.Dense(1024, activation='relu')(x) # 全连接层2
out_item = tf.keras.layers.Dense(len(item_label_names),
                                  activation='softmax',
                                  name='item')(x2) # 输出物品标签

# 创建模型
model = tf.keras.Model(inputs=inputs,
                       outputs=[out_color, out_item])

# print(model.summary()) # 打印模型结构

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss={'color': 'sparse_categorical_crossentropy',
                    'item': 'sparse_categorical_crossentropy'},
              metrics={'color': 'accuracy',
                       'item': 'accuracy'})


train_steps = train_count//BATCH_SIZE
test_steps = test_count//BATCH_SIZE
model_path = r'D:\Project\pythonProject\models\multi_output_model.h5'
def train():
    # 训练模型
    history = model.fit(train_ds,
                        epochs=4,
                        steps_per_epoch=train_steps,
                        validation_data=test_ds,
                        validation_steps=test_steps)
    # print(history.history) # 打印训练过程
    try:
        show_train_accuracy_info(history,"multi_output_model")
    except Exception as e:
        print(e)
    # 保存模型
    # model.save_weights(model_path) # 保存权重
    model.save(model_path) # 保存模型 # 会出现编码问题：{'black': 0, 'blue': 1,'red': 2} 变为 {'blue': 0, 'red': 1, 'black': 2}


# train()
# 加载模型
# model =model.load_weights(model_path)
model = tf.keras.models.load_model(model_path)

random_index = random.choice(range(len(all_image_paths)))
a_image_path = all_image_paths[random_index]
print(a_image_path)
my_image = load_image(a_image_path)
my_image = tf.expand_dims(my_image, axis=0)

pred = model.predict(my_image) # predict函数返回两个array列表，直接用model调用会返回两个Tensor列表列表
color_label = np.argmax(pred[0]) # 输出预测的标签
print(color_label_to_index)
print(color_label)
item_label = np.argmax(pred[1])
print(item_label_to_index)
print(item_label)
print(all_image_labels[random_index])