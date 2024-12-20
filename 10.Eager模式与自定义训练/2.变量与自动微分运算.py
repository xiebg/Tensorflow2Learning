import tensorflow as tf


"""
# 定义变量
v = tf.Variable(0.0)
print(v+1) # tf.Tensor(1.0, shape=(), dtype=float32)

print(v.assign(5)) # 改变变量的值，返回一个操作对象
print(v) # tf.Tensor(5.0, shape=(), dtype=float32)
print(v.assign_add(1)) # 增加变量的值，返回一个操作对象 tf.Tensor(6.0, shape=(), dtype=float32)

print(v.read_value()) # 读取变量的值，返回一个张量 tf.Tensor(6.0, shape=(), dtype=float32)

print("="*20)
# 求解梯度，自动微分运算变量,变量常量必须是float类型
w = tf.Variable(1.0)
with tf.GradientTape() as tape: # 记录梯度信息,自动跟踪变量运算
    loss = w*w # 导数为2w

gradient = tape.gradient(loss, w) # 计算梯度,求解loss对w的梯度导数
print(gradient) # tf.Tensor(2.0, shape=(), dtype=float32)

# 求解梯度，自动微分运算常量
w = tf.constant(3.0)
with tf.GradientTape() as tape: # 记录梯度信息,自动跟踪变量运算
    tape.watch(w) # 监视变量w
    loss = w*w # 导数为2w
d_loss_dw = tape.gradient(loss, w) # 计算梯度,求解loss对w的梯度导数，使用完会立即释放资源
print(d_loss_dw) # tf.Tensor(6.0, shape=(), dtype=float32)


print("="*20)
# 求解多个变量的梯度
w = tf.constant(3.0)
with tf.GradientTape(persistent=True) as tape: # persistent=True,可以多次调用tape.gradient()方法
    tape.watch(w) # 监视变量w
    y = w*w # 导数为2w
    z = y*y # 导数为2y

dy_dw = tape.gradient(y, w) # 计算梯度,求解y对w的梯度导数
dz_dw = tape.gradient(z, w) # 计算梯度,求解z对w的梯度导数
print(dy_dw) # tf.Tensor(6.0, shape=(), dtype=float32)
print(dz_dw) # tf.Tensor(108.0, shape=(), dtype=float32)
"""

print("="*20)
# 案例
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# print(train_images.shape) # (60000, 28, 28)
# 扩充维度，方便卷积运算
# train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
train_images = tf.expand_dims(train_images, axis=-1)  # (60000, 28, 28, 1)
test_images = tf.expand_dims(test_images, axis=-1)  # (10000, 28, 28, 1)

# 改变类型
# train_images = train_images.astype('float32') / 255.0
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
# loss_func = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True) # 小写的需要填写y_true, y_pred,

# 自定义训练
# features, labels = next(iter(dataset))
# print(features.shape) # (32, 28, 28, 1)
# print(labels.shape) # (32,)
# predictions = model(features) # (32, 10)
# print(predictions.shape) # (32, 10)
# print(tf.argmax(predictions, axis=-1)) # 2.3026

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

def train(model, dataset, epochs):
    for epoch in range(epochs): # 训练周期
        for (batch, (images, labels)) in enumerate(dataset): # 批次
            loss_value = train_step(model, images, labels)
            if batch % 100 == 0:
                print('Epoch {} Batch {} train_Loss {:.4f} train_Accuracy {:.4f}'.format(epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        print("==" * 20)
        for (batch, (images, labels)) in enumerate(test_dataset): # 批次
            test_step(model, images, labels)
            print('Epoch {} test_Loss {:.4f} test_Accuracy {:.4f}'.format(epoch + 1, test_loss.result(),test_accuracy.result()))
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()


train(model, dataset, 10) # 训练10轮

