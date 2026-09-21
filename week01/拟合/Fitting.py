import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 解决中文、负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成带噪声的 sin(x) 曲线数据（标准课堂实验数据）
np.random.seed(42)
X = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
y = np.sin(X) + np.random.normal(0, 0.15, size=X.shape)

# 2. 划分训练集、测试集
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()

# 创建画布，三图并列
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
for i in range(3):
    ax[i].scatter(X, y, c='yellow')

# ========== 1. 欠拟合：1次多项式（模型太简单） ==========
poly1 = PolynomialFeatures(degree=1)
x_train1 = poly1.fit_transform(trainX)
x_test1 = poly1.fit_transform(testX)
model.fit(x_train1, trainY)

train_loss1 = mean_squared_error(trainY, model.predict(x_train1))
test_loss1 = mean_squared_error(testY, model.predict(x_test1))

ax[0].plot(X, model.predict(poly1.fit_transform(X)), c='red')
ax[0].set_title("欠拟合（1次多项式）")
ax[0].text(-3, 1, f"测试误差：{test_loss1:.4f}")
ax[0].text(-3, 1.3, f"训练误差：{train_loss1:.4f}")

# ========== 2. 恰好拟合：5次多项式（匹配sin曲线规律） ==========
poly5 = PolynomialFeatures(degree=5)
x_train2 = poly5.fit_transform(trainX)
x_test2 = poly5.fit_transform(testX)
model.fit(x_train2, trainY)

train_loss2 = mean_squared_error(trainY, model.predict(x_train2))
test_loss2 = mean_squared_error(testY, model.predict(x_test2))

ax[1].plot(X, model.predict(poly5.fit_transform(X)), c='red')
ax[1].set_title("恰好拟合（5次多项式）")
ax[1].text(-3, 1, f"测试误差：{test_loss2:.4f}")
ax[1].text(-3, 1.3, f"训练误差：{train_loss2:.4f}")

# ========== 3. 过拟合：20次多项式（模型过于复杂） ==========
poly20 = PolynomialFeatures(degree=20)
x_train3 = poly20.fit_transform(trainX)
x_test3 = poly20.fit_transform(testX)
model.fit(x_train3, trainY)

train_loss3 = mean_squared_error(trainY, model.predict(x_train3))
test_loss3 = mean_squared_error(testY, model.predict(x_test3))

ax[2].plot(X, model.predict(poly20.fit_transform(X)), c='red')
ax[2].set_title("过拟合（20次多项式）")
ax[2].text(-3, 1, f"测试误差：{test_loss3:.4f}")
ax[2].text(-3, 1.3, f"训练误差：{train_loss3:.4f}")

plt.tight_layout()
plt.show()