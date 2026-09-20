import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.manifold import TSNE
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 统一从Excel读取数据（所有算法用同一份数据）
df = pd.read_excel("housing_data.xlsx")

# 取数值特征矩阵
X_feat = df[["area","house_age"]].values

# ↓ 下面接各自的算法代码 + 可视化

# t-SNE降维 - 房价场景
# 背景：t-SNE是非线性降维，擅长把高维数据的局部结构在低维（2维）展示出来，
# 适合可视化，一般不作为模型训练特征。
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(X_feat)
df["tsne_x"] = tsne_result[:,0]
df["tsne_y"] = tsne_result[:,1]

print("\nt-SNE降维完成（2维，用于可视化）")

# t-SNE专属可视化
plt.figure(figsize=(8,6))
plt.scatter(df["tsne_x"], df["tsne_y"], c=df["price"], cmap="viridis", alpha=0.7)
plt.colorbar(label="房价")
plt.xlabel("t-SNE维度1")
plt.ylabel("t-SNE维度2")
plt.title("t-SNE降维可视化（颜色=房价）")

plt.tight_layout()
plt.show()
