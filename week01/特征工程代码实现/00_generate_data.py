import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 500

area = np.random.normal(loc=100, scale=15, size=n_samples)
house_age = np.random.normal(loc=8, scale=3, size=n_samples)
district = np.random.choice(["城东","城西","城南"], size=n_samples)
price = area * 8000 - house_age * 2000 + np.random.normal(0, 5000, size=n_samples)

df = pd.DataFrame({"area":area,"house_age":house_age,
                   "district":district,"price":price})
df.to_excel("housing_data.xlsx", index=False)
print(" 数据已写入 housing_data.xlsx")