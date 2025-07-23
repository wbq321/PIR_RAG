import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("msmarco_data/corpus_10000.csv")

texts = df['text'].dropna().tolist()

byte_lengths = [len(t.encode('utf-8')) for t in texts]



# 打印基本统计信息

print(pd.Series(byte_lengths).describe())



# 绘制直方图

plt.hist(byte_lengths, bins=100, log=True) # 使用对数刻度可能看得更清楚

plt.title("Distribution of Document Byte Lengths")

plt.xlabel("Byte Length")

plt.ylabel("Frequency (log scale)")

plt.show()
