# -*- coding: utf-8 -*-

# kenerl scatter plot
import xarray as xr
import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def rmse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

# 读取 Excel 文件
Tallo = pd.read_csv('/stu01/xiangjy23/treedata/make_CrownRadius/vali/crdata_vali_avg_0517.csv')
# 提取两列数据并组成新表
df = Tallo.loc[:, ['cr_avg', 'cr_nc']]
# 将cd_nc列中每一个元素从 字符串类型 转变为 浮点类型
df['cr_nc'] = df['cr_nc'].astype(float)
# df['cd_nc'] = df['cd_nc'].apply(lambda x: x.strip('[]').split(', '))
# df['cd_nc'] = df['cd_nc'].astype(float)
# 删除包含 NaN 值的行
data = df.dropna()
# # 删除包含 [nan] 字符串的行
# data = data[~data['cd_nc'].astype(str).str.contains('nan')]

pft_lai = data['cr_avg'].values
grid_lai = data['cr_nc'].values


r2h = r2_score(pft_lai, grid_lai)
rmseh = rmse(pft_lai, grid_lai)
maeh = mae(pft_lai, grid_lai)

pft_lai = pft_lai.flatten()
grid_lai = grid_lai.flatten()
test = np.vstack([pft_lai, grid_lai])
density = gaussian_kde(test)(test)

idy = density.argsort()  # 通过 argsort 获取排序后的索引
pft_lai = pft_lai[idy]  # 使用排序后的索引重新排序 lai 和 grid_lai 数组
grid_lai = grid_lai[idy]
density = density[idy]
# 找到lai和grid_lai的最大值
max_value = max(pft_lai.max(), grid_lai.max())
# 对max_value向上取整
max_value = math.ceil(max_value)+1

fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)  # 绘制图表并添加指标文本
scatter = ax.scatter(pft_lai, grid_lai, marker='o', c=density, s=5, cmap='Spectral_r')
# ax.set_title("RandomForest Regression", fontsize=12)
ax.set_title("CR",fontsize=14)
ax.set_xlabel("observed CR (m)", fontsize=10)
ax.set_ylabel("predicted CR (m)", fontsize=10)
# 设置横纵坐标范围相同
ax.set_xlim(0, max_value)
ax.set_ylim(0, max_value)

# 设置坐标刻度和纵横比一致
ax.set_aspect('equal', adjustable='box', anchor='C')

# 设置坐标轴刻度一致
ax.set_xticks(np.arange(0, max_value, 2))
ax.set_yticks(np.arange(0, max_value, 2))

ax.text(0.85, 1.02, f"N = {len(pft_lai)}", transform=ax.transAxes, fontsize=10)

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = fig.colorbar(scatter, cax=cax, label='KDE')
# 设置线宽为0.8
ax.plot([0, max_value], [0, max_value], color='black', linestyle='-', linewidth=0.8)
ax.text(0.05, 0.9, f"R2: {r2h:.2f}", transform=ax.transAxes, fontsize=10)
ax.text(0.05, 0.84, f"RMSE: {rmseh:.2f}", transform=ax.transAxes, fontsize=10)
ax.text(0.05, 0.78, f"MAE: {maeh:.2f}", transform=ax.transAxes, fontsize=10)
plt.savefig(f"CR_vali_avg_0517.svg", dpi=300, format="svg")
