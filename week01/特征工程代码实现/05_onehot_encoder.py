import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
df = pd.read_excel("housing_data.xlsx")

# 导入对应算法
from sklearn.preprocessing import OneHotEncoder

# 独热编码（完整展示所有类别，无丢弃基准类）
ohe = OneHotEncoder(sparse_output=False)
onehot_res = ohe.fit_transform(df[["district"]])
onehot_df = pd.DataFrame(onehot_res, columns=["城东","城西","城南"])
df = pd.concat([df, onehot_df], axis=1)

print("地段完整独热编码结果：")
print(df[["district","城东","城西","城南"]].head())

# 【修复关键】每次绘图强制清空画布，杜绝图形叠加导致柱子等高
plt.figure(figsize=(8,5))
avg_price = df.groupby("district")["price"].mean()
# 手动固定顺序，保证三张图永远按 城东/城西/城南 展示
avg_price = avg_price.reindex(["城东","城西","城南"])
avg_price.plot(kind="bar", color=["#1f77b4","#ff7f0e","#2ca02c"])
plt.title("不同地段平均房价对比")
plt.ylabel("平均房价")
plt.tight_layout()
plt.show()