import os

import numpy as np
import tensorflow as tf

# 5. 自定义训练模型中的模型保存与恢复


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
    tf.keras.layers.Dense(10)
])

# 模型保存检查点的配置
model_Checkpoint_path = r'D:\Project\pythonProject\models\ckpt\save_checkpoint_model.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_Checkpoint_path, # 保存路径
                                                 save_weights_only=True, # 只保存模型权重
                                                 verbose=1) # 保存频率


optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y):
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 保存目录
cp_dir = r'D:\Project\pythonProject\models\ckpt_customize'
cp_prefix = os.path.join(cp_dir, 'ckpt') # 设置模型前缀
checkpoint = tf.train.Checkpoint( # 设置需要保存的内容
    optimizer=optimizer,
    model=model
)

def train_step(model, images, labels):
  with tf.GradientTape() as tape:
    pred = model(images)
    loss_value = loss(model, images, labels)
  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  train_loss(loss_value)
  train_accuracy(labels, pred)


dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(1000).batch(32)

def train():
    for epoch in range(5):
        for (batch,(images, labels)) in enumerate(dataset):
            train_step(model, images, labels)
            print('.', end='')
        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch+1, train_loss.result(), train_accuracy.result()))
        train_loss.reset_states()
        train_accuracy.reset_states()
        # 保存模型
        if (epoch+1) % 2 == 0: # 每2轮保存一次
            checkpoint.save(file_prefix=cp_prefix) # 保存模型


# train()

# 加载模型
new_point_path = tf.train.latest_checkpoint(cp_dir) # 获取最新检查点的模型路径
checkpoint.restore(new_point_path)
y_ = tf.argmax(model(test_images, training=False), axis=-1).numpy()

print("测试集准确率：",(y_ == test_labels).sum()/len(test_labels)) # 10000