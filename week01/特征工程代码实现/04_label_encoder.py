import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 统一从Excel读取数据（所有算法用同一份数据）
df = pd.read_excel("housing_data.xlsx")

# 取数值特征矩阵
X_feat = df[["area","house_age"]].values

# ↓ 下面接各自的算法代码 + 可视化

# 标签编码 - 房价场景
# 背景：城区 district 是文本（城东/城西/城南），模型无法直接读取，需要编码。
# 注意：LabelEncoder 会把无序城区编成 0/1/2，人为引入"城东<城西<城南"的虚假大小关系，
# 一般只用于目标标签，无序特征不推荐。
le = LabelEncoder()
df["district_label"] = le.fit_transform(df["district"])

print("\n标签编码结果（无序城区被编为0/1/2，引入虚假大小关系）：")
print(df[["district","district_label"]].head(10))

# 标签编码专属可视化
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
df["district"].value_counts().plot(kind="bar", color="steelblue")
plt.title("原始城区类别分布")
plt.ylabel("数量")

plt.subplot(1,2,2)
df["district_label"].value_counts().sort_index().plot(kind="bar", color="darkorange")
plt.title("LabelEncoder编码后（0/1/2）")
plt.ylabel("数量")

plt.tight_layout()
plt.show()
