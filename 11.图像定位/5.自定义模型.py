import tensorflow as tf

# 继承tf.keras.Model类，实现自己的模型
# 自定义层与模型的区别：./img/13
# 使用和tf.keras.Layer相同的API，可以方便地迁移到tf.keras.Model
class MLP_model(tf.keras.Model):  # 多层感知机
    def __init__(self):
        super(MLP_model, self).__init__()
        self.lin_1 = tf.keras.layers.Dense(32)
        self.lin_2 = tf.keras.layers.Dense(64)
        self.lin_3 = tf.keras.layers.Dense(32)

    def call(self, inputs):
        x1 = self.lin_1(inputs)  # 调用lin_1层
        x1 = tf.nn.relu(x1)  # 激活函数
        x2 = self.lin_2(x1)  # 调用lin_2层
        x2 = tf.nn.relu(x2)  # 激活函数
        x3 = self.lin_3(x2)  # 调用lin_3层
        return tf.concat([x1, x3])  # 连接输出


model = MLP_model()  # 实例化模型
print(model.summary())  # 打印模型结构
print(model.fit())
print(model.predict())
print(model.save())
