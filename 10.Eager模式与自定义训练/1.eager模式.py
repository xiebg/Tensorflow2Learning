import tensorflow as tf
import numpy as np

print(tf.__version__)
print(tf.executing_eagerly())  # True 表示当前处于eager模式

# 1. eager模式
# 2. eager模式下，TensorFlow会立即执行操作，而非构建计算图，可以立即得到结果。
x = [[2, ]]
m = tf.matmul(x, x)  # 矩阵乘法
print(m)  # [[4]]
print(m.numpy())  # [[4]]

# 建立常量
a = tf.constant([[1, 2], [3, 4]])
b = tf.add(a, 1)  # [[2 3] [4 5]] # 常量相加
c = tf.matmul(a, b)  # [[10 13][22 29]] # 矩阵乘法

print(c)

# 使用的是python控制流，而不是图结构。
# 因此，在eager模式下，可以更方便地进行调试和开发。
num = tf.convert_to_tensor(10)  # 将10转换为Tensor
for i in range(num.numpy()):
    i = tf.constant(i)
    if int(i % 2) == 0:
        print("Even number: ", i.numpy())
    else:
        print("Odd number: ", i.numpy())

d = np.array([[1, 2], [3, 4]])
print(a + d)  # tf.Tensor([[2 4][6 8]], shape=(2, 2), dtype=int32)
print((a + d).numpy())

g = tf.convert_to_tensor(10)
print(float(g))
