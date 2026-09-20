import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
df = pd.read_excel("housing_data.xlsx")

# 仅对【房价】做对数变换（核心修正：面积无偏态，无需变换）
df["price_log"] = np.log1p(df["price"])

print("对数变换前后房价对比：")
print(df[["price","price_log"]].head())

# 可视化
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.hist(df["price"], bins=30, alpha=0.7, color="crimson")
plt.title("原始房价分布（右偏）")

plt.subplot(1,2,2)
plt.hist(df["price_log"], bins=30, alpha=0.7, color="purple")
plt.title("对数变换后房价分布（趋近正态）")

plt.tight_layout()
plt.show()