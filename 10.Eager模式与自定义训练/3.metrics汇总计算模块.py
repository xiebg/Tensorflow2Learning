import tensorflow as tf
import numpy as np
"""
m = tf.keras.metrics.Mean("acc") # 定义均值计算器
m([1, 2, 3, 4]) # 计算均值,为2.5
m([5, 6, 7, 8]) # 计算均值，为(6.5+2.5)/2=4.5
print(m.result().numpy()) # 输出均值

m.reset_states() # 重置均值计算器
print(m.result().numpy()) # 输出均值，为0.0
"""

a = tf.keras.metrics.SparseCategoricalAccuracy("acc") # 定义稀疏分类准确率计算器) # 定义稀疏分类准确率计算器