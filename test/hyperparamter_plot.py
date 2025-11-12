import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

cifar10_data = {
    'beta': np.array([8, 10, 12, 14, 16, 18, 20]),
    'sym0.9': np.array([90.97, 91.16, 90.2, 89.17, 90.97, 89.42, 89.50]),
    'sym0.8': np.array([94.54, 94.58, 94.59, 94.09, 94.03, 94.25, 93.81]),
    'sym0.5': np.array([95.53, 95.6, 95.4, 95.74, 95.7, 95.4, 95.67])
}

cifar100_data = {
    'beta': np.array([150, 200, 250, 300, 350, 400, 450]),
    'sym0.9': np.array([61.05, 61.22, 61.67, 61.32, 61.72, 61.28, 59.26]),
    'sym0.8': np.array([72.42, 72.55, 72.65, 72.64, 72.70, 72.60, 72.50]),
    'sym0.5': np.array([78.31, 77.96, 77.84, 77.86, 77.67, 77.47, 77.41])
}

styles = {
    'sym0.9': {'color': '#2ca02c', 'marker': 'v', 'label': 'sym=0.9'}, # Green
    'sym0.8': {'color': '#d62728', 'marker': 'v', 'label': 'sym=0.8'}, # Red
    'sym0.5': {'color': '#9467bd', 'marker': 'v', 'label': 'sym=0.5'}  # Purple
}

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']

fig1, ax1 = plt.subplots(figsize=(4, 3))

ax1.plot(cifar10_data['beta'], cifar10_data['sym0.9'], **styles['sym0.9'])
ax1.plot(cifar10_data['beta'], cifar10_data['sym0.8'], **styles['sym0.8'])
ax1.plot(cifar10_data['beta'], cifar10_data['sym0.5'], **styles['sym0.5'])

# 设置更大字号的标题和标签
ax1.set_title('CIFAR-10 (alpha=1)', fontsize=20, pad=10)
ax1.set_xlabel('beta', fontsize=20)
ax1.set_ylabel('Accuracy', fontsize=20)

# 设置更大字号的刻度
ax1.tick_params(axis='both', which='major', labelsize=16)
ax1.grid(True, linestyle='--', alpha=0.6)

ax1.set_xticks(cifar10_data['beta'])

fig1.savefig('cifar10_plot.pdf', dpi=300, bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(4, 3))

ax2.plot(cifar100_data['beta'], cifar100_data['sym0.9'], **styles['sym0.9'])
ax2.plot(cifar100_data['beta'], cifar100_data['sym0.8'], **styles['sym0.8'])
ax2.plot(cifar100_data['beta'], cifar100_data['sym0.5'], **styles['sym0.5'])

ax2.set_title('CIFAR-100 (alpha=2)', fontsize=20, pad=10)
ax2.set_xlabel('beta', fontsize=20)
ax2.set_ylabel('Accuracy', fontsize=20)


ax2.tick_params(axis='both', which='major', labelsize=16)
ax2.grid(True, linestyle='--', alpha=0.6)

# 移除图例
# ax2.legend()
ax2.set_xticks(cifar100_data['beta'][::2])

fig2.savefig('cifar100_plot.pdf', dpi=300, bbox_inches='tight')
plt.close(fig2)

legend_elements = [
    Line2D([0], [0], color=styles['sym0.9']['color'], marker=styles['sym0.9']['marker'], markersize=8, linestyle='-', label=styles['sym0.9']['label']),
    Line2D([0], [0], color=styles['sym0.8']['color'], marker=styles['sym0.8']['marker'], markersize=8, linestyle='-', label=styles['sym0.8']['label']),
    Line2D([0], [0], color=styles['sym0.5']['color'], marker=styles['sym0.5']['marker'], markersize=8, linestyle='-', label=styles['sym0.5']['label'])
]

# 调整 figsize 以适应更大的字体
fig_legend = plt.figure(figsize=(8, 0.6))
# 设置图例字体大小
legend = fig_legend.legend(handles=legend_elements, loc='center', ncol=3, edgecolor='lightgray', fontsize=18)

ax_legend = fig_legend.gca()
ax_legend.axis('off')

fig_legend.savefig('legend.pdf', dpi=300, bbox_inches='tight')
plt.close(fig_legend)