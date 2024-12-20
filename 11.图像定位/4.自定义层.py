import tensorflow as tf


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):  # 初始化,units为输出维度,input_dim为输入维度
        super(Linear, self).__init__()
        # 定义模型参数
        w_init = tf.random_normal_initializer()  # 标准正态分布初始化
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units),
                                 dtype='float32'),
            trainable=True  # 是否参与训练，默认True
        )

        b_init = tf.zeros_initializer()  # 全零初始化
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):  # 定义前向传播，矩阵乘法和加法
        return tf.matmul(inputs, self.w) + self.b


my_linear = Linear(units=4, input_dim=2)  # 实例化自定义层
x = tf.ones((2, 2))  # 输入,2行2列0
y = my_linear(x)  # 前向传播

print(my_linear.w)  # 输出权重
print(my_linear.b)  # 输出偏置
print(my_linear.weights)  # 输出权重和偏置列表
print(my_linear.trainable_variables)  # 输出可训练变量列表


# 更快捷的方式为层添加权重：add_weight()方法
class Linear2(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):  # 初始化,units为输出维度,input_dim为输入维度
        super(Linear2, self).__init__()
        # 定义模型参数
        self.w = self.add_weight(
            shape=(input_dim, units),
            initializer='random_normal',
            trainable=True)

        self.b = self.add_weight(
            shape=(units,),
            initializer='zeros',
            trainable=True)

    def call(self, inputs):  # 定义前向传播，矩阵乘法和加法
        return tf.matmul(inputs, self.w) + self.b


# 将权重的创建推迟到得知输入形状之后，build()方法中,原理是通过__call__()方法调用build()方法
class Linear3(tf.keras.layers.Layer):
    def __init__(self, units=32):  # 初始化,units为输出维度,input_dim为输入维度
        super(Linear3, self).__init__()
        # 定义模型参数
        self.units = units

    def build(self, input_shape):  # input_shape为输入形状(batch_size, input_dim)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True)

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True)

    def call(self, inputs):  # 定义前向传播，矩阵乘法和加法
        return tf.matmul(inputs, self.w) + self.b


my_linear3 = Linear3(4)  # 实例化自定义层
print(my_linear3.weights)  # 输出权重和偏置列表,[]
y = my_linear3(x)  # 前向传播
print(my_linear3.weights)  # 输出权重和偏置列表,[(<tf.Variable 'linear3/kernel:0' shape=(2, 4) dtype=float32, numpy=...


# 层的可递归组合
# 如果将一个层实例分配给另一个层的特性，则外部层将开始跟踪内部层的权重。
# 建议在init()方法中创建此类层（由于子层通常具有构建方法，它们将与外部层一起构建），在call()方法中调用子层。
class MLPBlock(tf.keras.layers.Layer):  # 多层感知机
    def __init__(self):
        super(MLPBlock, self).__init__()
        self.lin_1 = Linear3(32)
        self.lin_2 = Linear3(64)
        self.lin_3 = Linear3(1)

    def call(self, inputs):
        x = self.lin_1(inputs)  # 调用lin_1层
        x = tf.nn.relu(x)  # 激活函数
        x = self.lin_2(x)  # 调用lin_2层
        x = tf.nn.relu(x)  # 激活函数
        x = self.lin_3(x)  # 调用lin_3层
        return x


mlp = MLPBlock()  # 实例化自定义层
mlp((None, 2))  # 构建层,输入形状(batch_size, input_dim)


class MLPBlock2(tf.keras.layers.Layer):  # 多层感知机
    def __init__(self):
        super(MLPBlock2, self).__init__()
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

