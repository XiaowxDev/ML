import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 统一从Excel读取数据（所有算法用同一份数据）
df = pd.read_excel("housing_data.xlsx")

# 取数值特征矩阵
X_feat = df[["area","house_age"]].values

# ↓ 下面接各自的算法代码 + 可视化

# MinMax归一化 - 房价场景专属代码
scaler_minmax = MinMaxScaler()
# 对面积、房龄、房价特征归一化
df[["area_minmax","age_minmax","price_minmax"]] = scaler_minmax.fit_transform(df[["area","house_age","price"]])

print("\n归一化前后对比（房屋面积）：")
print(df[["area","area_minmax"]].head())

# 归一化专属可视化
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(df["area"], bins=30, alpha=0.7, color="steelblue")
plt.title("原始房屋面积分布（尺度大）")
plt.xlabel("面积")

plt.subplot(1,2,2)
plt.hist(df["area_minmax"], bins=30, alpha=0.7, color="darkorange")
plt.title("MinMax归一化后面积分布（0~1）")
plt.xlabel("归一化面积")

plt.tight_layout()
plt.show()