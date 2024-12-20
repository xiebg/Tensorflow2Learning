import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 输入[batch_size, time_steps, input_dim],[批次大小，序列长度，输入维度]
# 输出[batch_size,output_dim],[批次大小，输出维度]

# 航空公司评论数据集

data = pd.read_csv('./data/Tweets.csv')
data = data[['airline_sentiment', 'text']]
# print(data.airline_sentiment.unique())  # ['neutral' 'positive' 'negative']
# print(data.airline_sentiment.value_counts())  # negative(差评):9178 neutral(中性):3099 positive(好评):2363
data_p = data[data.airline_sentiment == 'positive']  # 好评 2363
data_n = data[data.airline_sentiment == 'negative']  # 差评 9178
# data_u = data[data.airline_sentiment == 'neutral']  # 中性

data_n = data_n.iloc[:len(data_p)]  # 取出差评数据中前2363条数据
data = pd.concat([data_p, data_n])  # 合并好评和差评数据

data = data.sample(len(data))  # 打乱数据
data['review'] = (data.airline_sentiment == 'positive').astype(int)  # 好评为1，差评为0
data = data[['text', 'review']]  # 选择text和review列
# print(data.head())

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



# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_word, 128, input_length=max_len),  # 词嵌入层,把文本转换为密集向量（类似one-hot编码），128维
    tf.keras.layers.LSTM(64),  # LSTM层，64维，return_sequences=True表示输出每个时间步的隐藏状态,LSTM层只需要一层即可，不能再加层
    tf.keras.layers.Dense(1, activation='sigmoid')  # 输出层
])

# model.summary()

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(text_data,text_label,
                    epochs=10,
                    batch_size=64,  # 批次大小
                    validation_split=0.2
                    ) # 验证集比例