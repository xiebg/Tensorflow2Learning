import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

incomedata = pd.read_csv('./data/income.csv')
x = incomedata.Education
y = incomedata.Income

def plot_data(x,y):
    plt.scatter(x,y)
    plt.show()
# plot_data(x,y)

def model(x,y):
    # 初始化模型，空的
    model = tf.keras.Sequential()  # Sequential 序列模型，可以按顺序添加层 Sequential：连续式；顺序；顺序语句；序列；循序
    # 加载第一层模型Dense, y = ax+b
    model.add(
        tf.keras.layers.Dense(1,input_shape=(1,)) # Dense 全连接层，输入1维(None,1)，输出1维
    ) #形状：输出1维，输入(1,)
    # 打印模型形状
    print(model.summary()) # Param 2 表示初始化a和b两个参数


    # 编译（配置）
    model.compile(
        optimizer='adam',   # 优化方法，adam是一种自适应优化算法
        loss='mse'    # 损失函数是梯度下降，mse是均方误差
    )

    # 训练
    history = model.fit(x,y,epochs=5000)
    print(history)
    #
    # 预测
    pre1 = model.predict(x)
    pre2 = model.predict(pd.Series([20]))  # 预测20
    print(pre1)
    print(pre2)

model(x,y)