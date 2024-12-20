import numpy as np
import tensorflow as tf

# 1. 模型整体保存
# 2. 保存模型框架（代码结构）
# 3. 保存模型权重（参数）
# 4. 使用回调函数保存模型
# 5. 自定义训练循环的模型保存
# * 整个路径不能有中文，否则会报错

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = np.expand_dims(train_images, axis=-1) # -1表示最后一个维度扩张
test_images = np.expand_dims(test_images, axis=-1)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=train_images.shape[1:],padding='same'),   # (28, 28, 1)
    tf.keras.layers.MaxPooling2D(),     # 池化层,默认池化窗口大小为(2,2),增加卷积层的视野范围（14，14，32）
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # (12, 12, 64)，填充模式为valid
    tf.keras.layers.GlobalAveragePooling2D(), # 全局平均池化层（将所有维度的特征图的平均值作为输出），将(None,12, 12, 64)转换为(None, 64)，减少参数数量。Flatten缺点是丢失了空间信息，并且引入了过多的计算参数
    tf.keras.layers.Dense(10, activation='softmax')
])

model_path = r'D:\Project\pythonProject\models\save_all_model.h5'
model_json_path = r'D:\Project\pythonProject\models\save_json_model.json'
model_weights_path = r'D:\Project\pythonProject\models\save_weights_model.h5'

def save_train_model():
    # 编译模型
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # 训练模型
    history = model.fit(train_images, train_labels,
                        epochs=2,
                        validation_data=(test_images, test_labels))

    model.evaluate(test_images, test_labels,verbose=0) # 评估模型, evaluate()方法返回loss和metrics的平均值
    # verbose=0表示不输出评估结果,1表示输出进度条,2表示每个epoch输出一次评估结果

    # # 1. 模型整体保存,包含模型框架和权重，优化器的配置等信息，无需访问原始代码即可恢复模型继续训练,keras使用HDF5标准格式保存模型
    # model.save(model_path) # 保存模型

    # # 2. 仅保存模型框架（代码结构）
    # model_json = model.to_json()
    # with open(model_json_path, 'w') as f:
    #     f.write(model_json)

    # 3. 仅保存模型权重（参数）
    # weights = model.get_weights()
    # np.save(save_weights_model.npy, weights)
    model.save_weights(model_weights_path) # 保存模型权重

# save_train_model()

# 加载模型
# # 1. 加载模型整体
# new_model = tf.keras.models.load_model(model_path)
# # new_model.summary() # 打印模型结构
# print(new_model.evaluate(test_images, test_labels,verbose=0)) # 评估模型

# 2. 加载模型框架（代码结构）
with open(model_json_path, 'r') as f:
    model_json = f.read()
reinitialized_model = tf.keras.models.model_from_json(model_json)
print(reinitialized_model.summary()) # 打印模型结构
reinitialized_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
print(reinitialized_model.evaluate(test_images, test_labels,verbose=0)) # 评估模型,随机权重

# 3. 加载模型权重（参数）,结合方法2一起使用
reinitialized_model.load_weights(model_weights_path) # 加载模型权重
print(reinitialized_model.evaluate(test_images, test_labels,verbose=0)) # 评估模型,加载权重后模型性能提升
# 方法2+方法3不等于方法1，没有保存优化器配置信息

