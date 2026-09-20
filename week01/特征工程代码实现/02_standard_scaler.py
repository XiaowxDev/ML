import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 统一从Excel读取数据（所有算法用同一份数据）
df = pd.read_excel("housing_data.xlsx")

# 取数值特征矩阵
X_feat = df[["area","house_age"]].values

# ↓ 下面接各自的算法代码 + 可视化

# Z-Score标准化 - 房价场景
# 背景：逻辑回归/SVM/PCA等算法假设输入特征近似正态分布，
# 标准化后每个特征均值为0、方差为1，消除量纲差异。
scaler_std = StandardScaler()
df[["area_std","age_std","price_std"]] = scaler_std.fit_transform(df[["area","house_age","price"]])

print("\n标准化前后对比（房屋面积）：")
print(df[["area","area_std"]].head())

# 标准化专属可视化
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df["area"], bins=30, alpha=0.7, color="steelblue")
plt.title("原始房屋面积分布")
plt.xlabel("面积")

plt.subplot(1,2,2)
plt.hist(df["area_std"], bins=30, alpha=0.7, color="darkorange")
plt.title("Z-Score标准化后（均值0，方差1）")
plt.xlabel("标准化面积")

plt.tight_layout()
plt.show()
