import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('./data/credit-a.csv')

x = data.iloc[:, 1:-1]
y = data.iloc[:, -1]

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'), # 输入层,10个神经元,输入维度为3
    tf.keras.layers.Dense(1) # 输出层,1个神经元
])
print(model.summary())

# 编译模型
model.compile(optimizer='adam', loss='mse')


# 训练模型
history = model.fit(x, y, epochs=5000)

# 绘制损失函数
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# 预测结果
y_pred = model.predict(x[:10, :]) # 预测前10个样本的结果
print(y_pred)