import numpy as np
import tensorflow as tf

# 4. 在训练期间保存模型检查点
# tf.keras.callbacks.ModelCheckpoint(filepath=model_path, # 保存路径
#                                    monitor='val_loss', # 监控的指标,val_loss表示验证集的loss
#                                    save_best_only=True, # 是否只保存最佳模型,默认为False，当为True时，只保存loss最低时的模型权重，与monitor配合使用
#                                    save_weights_only=False, # 是否只保存模型权重，默认为False，当为True时，只保存模型权重，不保存整个模型
#                                    mode='auto', # 当监控指标为val_loss时，mode='auto'表示当val_loss下降时保存模型，当监控指标为val_acc时，mode='auto'表示当val_acc上升时保存模型
#                                    save_freq='epoch') # 保存频率，默认为epoch，表示每个epoch保存一次模型，也可以设置为steps，表示每多少步保存一次模型

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

# 模型保存检查点的配置
model_Checkpoint_path = r'D:\Project\pythonProject\models\ckpt\save_checkpoint_model.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_Checkpoint_path, # 保存路径
                                                 save_weights_only=True, # 只保存模型权重
                                                 verbose=1) # 保存频率

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
def save_train_model():
    # 训练模型
    history = model.fit(train_images, train_labels,
                        epochs=2,
                        callbacks=[cp_callback], # 使用回调函数保存模型
                        validation_data=(test_images, test_labels))




# save_train_model()
print(model.evaluate(test_images, test_labels,verbose=0))
# 加载模型
model.load_weights(model_Checkpoint_path)
# 评估模型
print(model.evaluate(test_images, test_labels,verbose=0))


