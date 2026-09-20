import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
df = pd.read_excel("housing_data.xlsx")

# 三维原始输入特征：面积、房龄、房价
X_feat = df[["area", "house_age", "price"]].values

# ========== 1、完整三维PCA（保留全部主成分，3D可视化） ==========
from sklearn.decomposition import PCA
# 保留3个主成分，完成三维特征空间重构
pca_3d = PCA(n_components=3, random_state=42)
X_pca_3d = pca_3d.fit_transform(X_feat)

# 存储三维主成分
df["pca1"] = X_pca_3d[:, 0]
df["pca2"] = X_pca_3d[:, 1]
df["pca3"] = X_pca_3d[:, 2]

print("【三维PCA方差解释率】")
print(pca_3d.explained_variance_ratio_)

# 三维特征3D可视化
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(df["pca1"], df["pca2"], df["pca3"], c=df["price"], cmap="viridis", alpha=0.7)
ax.set_title("PCA三维特征空间分布（面积+房龄+房价）")
ax.set_xlabel("主成分1")
ax.set_ylabel("主成分2")
ax.set_zlabel("主成分3")
plt.colorbar(scatter, label="房价")
plt.tight_layout()
plt.show()

# ========== 2、PCA维度压缩（三维降一维，仅保留主成分1） ==========
pca_1d = PCA(n_components=1, random_state=42)
X_pca_1d = pca_1d.fit_transform(X_feat)
df["pca_main"] = X_pca_1d

print(f"\n【一维压缩主成分方差解释率】{pca_1d.explained_variance_ratio_[0]:.4f}")
print("\n压缩后一维主成分特征前5行：")
print(df[["pca_main"]].head())

# 一维主成分可视化
plt.figure(figsize=(10,5))
plt.scatter(df["pca_main"], df["price"], c=df["price"], cmap="viridis", alpha=0.7)
plt.title("PCA三维特征压缩至一维主成分")
plt.xlabel("压缩后主成分1（综合特征）")
plt.ylabel("原始房价")
plt.colorbar(label="房价梯度")
plt.tight_layout()
plt.show()