import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# 在音频生成和机器翻译领域取得巨大成功
# 文本分类和时间序列预测等简单任务，小型的一维卷积可以取代RNN
# 具有平移不变性
# 形状：(samples, time, feature)

# 航空公司评论数据集


data = pd.read_csv('./data/Tweets.csv')
data = data[['airline_sentiment', 'text']]

data_p = data[data.airline_sentiment == 'positive']  # 好评 2363
data_n = data[data.airline_sentiment == 'negative']  # 差评 9178

data_n = data_n.iloc[:len(data_p)]  # 取出差评数据中前2363条数据
data = pd.concat([data_p, data_n])  # 合并好评和差评数据

data = data.sample(len(data))  # 打乱数据
data['review'] = (data.airline_sentiment == 'positive').astype(int)  # 好评为1，差评为0
data = data[['text', 'review']]  # 选择text和review列

# tf.keras.layers.Embedding 把文本转换为向量
import re
token = re.compile('[A-Za-z]+|[!?.,;]')
def reg_rext(text):
    new_text = token.findall(text)
    new_text = [word.lower() for word in new_text]
    return new_text

data['text'] = data.text.apply(reg_rext)
# print(data.head())

# 每个单词用一个索引表示
word_set = set()
for text in data.text:
    for word in text:
        word_set.add(word)

print(len(word_set))  # 7099


word_index = {word: index+1 for index, word in enumerate(word_set)}  # 索引从1开始
index_word = {index: word for word, index in word_index.items()}

data['text'] = data.text.apply(lambda text: [word_index[word] for word in text])
max_len = max(len(text) for text in data.text)   # 39 最长评论长度
max_word = len(word_set) + 1  # 7100 词汇量大小

text_data = tf.keras.preprocessing.sequence.pad_sequences(data.text.values, maxlen=max_len)  # 填充0
text_label = data.review.values

def show_train_loss_info(history, name):
    import os
    if not os.path.exists('./image'):
        os.mkdir('./image')
    plt.plot(history.epoch, history.history['loss'], label='loss')
    plt.plot(history.epoch, history.history['val_loss'], label='val_loss')
    plt.legend()  # 显示图例
    plt.savefig(f'./image/{name}.png')
    plt.show()

# 一维卷积
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(max_word, 50, input_length=max_len))  # 词向量,50维
model.add(tf.keras.layers.Conv1D(32, 7, activation='relu',padding='same'))  # 卷积核大小7X1,32个
model.add(tf.keras.layers.MaxPooling1D(3))  # 序列长度缩减3倍
# model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Conv1D(32, 7, activation='relu',padding='same'))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer=tf.keras.optimizers.RMSprop(), # 优化器,RMSprop（Root Mean Square Propagation）“均方根传播法”
              loss='binary_crossentropy',   # 损失函数
              metrics=['accuracy']) # 评估指标

history = model.fit(text_data, text_label,
                    epochs=30,
                    batch_size=128,
                    validation_split=0.2)  # 训练模型


show_train_loss_info(history, 'conv1d')