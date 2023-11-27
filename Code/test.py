import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

# 生成一些示例数据
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([2, 3, 3.5, 4, 4.5])

# 使用LOWESS进行拟合
lowess = sm.nonparametric.lowess(y_data, x_data, frac=0.5)

# 获取拟合的结果
fitted_values = lowess[:, 1]

# 计算斜率（可以选择使用差分）
slopes = np.diff(fitted_values) / np.diff(x_data)

# 最新数据点的斜率
latest_slope = slopes[-1]

# 打印结果
print(f"Latest Slope (最新数据点的斜率): {latest_slope}")

# 绘制原始数据和拟合曲线
plt.scatter(x_data, y_data, label="原始数据")
plt.plot(x_data, fitted_values, color='red', label="拟合曲线")
plt.legend()
plt.show()
