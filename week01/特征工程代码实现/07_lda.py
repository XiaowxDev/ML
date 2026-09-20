import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
df = pd.read_excel("housing_data.xlsx")
X_feat = df[["area","house_age"]].values

# 导入对应算法
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# 房价离散分组，适配LDA分类降维
price_bin = pd.cut(df["price"], bins=3, labels=[0,1,2])

# LDA降维
lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_feat, price_bin)
df["lda_feature"] = X_lda

print("LDA降维特征前5行：")
print(df["lda_feature"].head())

# 可视化
plt.figure(figsize=(8,5))
labels = ["低价","中价","高价"]
for i in range(3):
    mask = (price_bin == i)
    plt.scatter(df.loc[mask,"lda_feature"], df.loc[mask,"price"], label=labels[i], alpha=0.7)
plt.legend()
plt.title("LDA有监督降维分层效果")
plt.xlabel("LDA降维特征")
plt.ylabel("房价")
plt.show()