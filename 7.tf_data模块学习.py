import tensorflow as tf

# # 创建方法1：从列表创建数据集
# # 1.创建数据集
# dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# # 2.创建迭代器
# for element in dataset:
#     print(element.numpy())  # 转化为numpy数组输出，2.0专用


# dataset1 = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]) # 每个组件的shape必须一致
# print(dataset1) # shape=(2,)表示每个组件的shape为2
# for element in dataset1:
#     print(element.numpy())


# dataset_dict = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
# print(dataset_dict)
# for element in dataset_dict:
#     print(element)
#     # {'a': 1, 'b': 4, 'c': 7}
#     # {'a': 2, 'b': 5, 'c': 8}
#     # {'a': 3, 'b': 6, 'c': 9}


# # 创建方法2：从numpy数组创建数据集
# import numpy as np
# dataset_np = tf.data.Dataset.from_tensor_slices(np.arange(10))
# # 随机打乱数据集
# dataset_np = dataset_np.shuffle(10)
# # 重复3次,共有30个元素,为None时表示无限重复
# dataset_np = dataset_np.repeat(count=3)
# # 批量大小为3，每次取3个元素
# dataset_np = dataset_np.batch(batch_size=3) # 每个batch有3个元素
#
# # for element in dataset_np.take(5): # 取前5个元素
# #     print(element)
#
# for element in dataset_np:
#     print(element.numpy())


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 使用函数对数据做变换
# dataset = dataset.map(lambda x: x * 2) #对每个元素进行乘2操作
dataset = dataset.map(tf.square) #对每个元素进行平方操作

for element in dataset:
    print(element.numpy())