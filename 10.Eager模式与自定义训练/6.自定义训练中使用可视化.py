import datetime

import tensorflow as tf


(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 扩充维度，方便卷积运算
train_images = tf.expand_dims(train_images, axis=-1)  # (60000, 28, 28, 1)
test_images = tf.expand_dims(test_images, axis=-1)  # (10000, 28, 28, 1)

# 改变类型
train_images = tf.cast(train_images, tf.float32) / 255.0
test_images = tf.cast(test_images, tf.float32) / 255.0
train_labels = tf.cast(train_labels, tf.int64)
test_labels = tf.cast(test_labels, tf.int64)

# 构建数据集
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
dataset = dataset.shuffle(1000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(32)
# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(None, None, 1)), # 卷积层
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10) # softmax函数只是对最后一层的输出做归一化，并不影响模型的预测结果，概率值最大的标签即为预测结果
])

optimizer = tf.keras.optimizers.Adam()
loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) # 可调用对象

def loss(model, x, y):
    y_ = model(x)
    return loss_func(y_true=y, y_pred=y_)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


def train_step(model,images, labels):
    # 追踪训练过程，记录梯度信息
    with tf.GradientTape() as tape:
        pred = model(images)
        loss_value = loss(model, images, labels)
    grads = tape.gradient(loss_value, model.trainable_variables)    # 梯度， model.trainable_variables 获取模型中所有可训练的变量
    optimizer.apply_gradients(zip(grads, model.trainable_variables)) # 梯度下降更新参数 optimizer.apply_gradients()方法更新模型参数
    train_loss(loss_value)
    train_accuracy(labels, pred)
    return loss_value

def test_step(model,images, labels):
    pred = model(images)
    loss_value = loss(model, images, labels)
    test_loss(loss_value)
    test_accuracy(labels, pred)
    return loss_value

# 5.模型可视化
import os
log_dir = os.path.join("D:\Project\pythonProject\logs\gradint_tape", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))   # 目录路径中不能存在中文
train_log_dir = os.path.join(log_dir, 'train')
test_log_dir = os.path.join(log_dir, 'test')
# （2）.创建写入器
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)


def train(model, dataset, epochs):
    for epoch in range(epochs): # 训练周期
        for (batch, (images, labels)) in enumerate(dataset): # 批次
            loss_value = train_step(model, images, labels)
            if batch % 100 == 0:
                print('Epoch {} Batch {} train_Loss {:.4f} train_Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        with train_summary_writer.as_default(): # 训练日志
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
        print("==" * 20)
        for (batch, (images, labels)) in enumerate(test_dataset): # 批次
            test_step(model, images, labels)
            print('Epoch {} test_Loss {:.4f} test_Accuracy {:.4f}'.format(epoch + 1, test_loss.result(),test_accuracy.result()))
        with test_summary_writer.as_default(): # 测试日志
            tf.summary.scalar('loss', test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


train(model, dataset, 10) # 训练10轮
# 启动tensorboard
# tensorboard --logdir=D:\Project\pythonProject\logs\gradint_tape\20241129-231555