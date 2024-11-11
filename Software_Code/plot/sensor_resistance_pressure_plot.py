import matplotlib.pyplot as plt
import numpy as np

# 原始数据
pressure = [2, 5.7, 11.9, 20, 29.9, 40, 60, 71.6, 80.3, 100, 111, 122, 131, 140, 150, 160]
R = [15.7, 9.05, 4.92, 2.79, 2.41, 2.15, 2.06, 2.02, 1.99, 1.9, 1.86, 1.81, 1.78, 1.76, 1.75, 1.75]

# 计算压力 (kPa)
area = np.pi * (0.01595) ** 2
pressure = np.array(pressure) / area * 0.001  # 转换为 kPa

# 定义最后一个 y 值
y_horizontal_line = R[-1]

# 创建图表
plt.figure(figsize=(8, 6))

# 绘制背景矩形区域
# 0 kPa - 20 kPa 背景颜色
plt.axvspan(0, 25, color='lightgoldenrodyellow', alpha=0.5, label='0-25 kPa Region')

# 20 kPa - 200 kPa 背景颜色
plt.axvspan(25, 200, color='lightsteelblue', alpha=0.3, label='25-200 kPa Region')

# 绘制折线图，线宽4，点的大小设置为10
plt.plot(pressure, R, marker='o', markersize=10, linewidth=4, linestyle='-', color='mediumblue', label='Sensor Resistance vs Pressure curve')

# 水平虚线，线宽3
plt.axhline(y=y_horizontal_line, linewidth=4, color='slategrey', linestyle='--', label= 'Sensor Resistance Saturation')

# 设置标题和轴标签，字体加大加粗
plt.title('Sensor Resistance vs Pressure', fontsize=30, fontweight='bold')
plt.xlabel('Pressure (kPa)', fontsize=22, fontweight='bold')
plt.ylabel('Sensor Resistance (kOhm)', fontsize=22, fontweight='bold')



plt.yscale('log')


# 设置轴的线条加粗
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['bottom'].set_linewidth(2)

# 设置刻度参数的字体大小
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)

# 添加网格
plt.grid(True)

# 显示图例
plt.legend(fontsize = 17 ,loc="upper right")

# 显示图表
plt.show()


# 空数组，用于存放在硬性材料上的数据（x点）
# pressure_data_hard_material = [20,  30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]  # x轴数据：压强
# sensor_resistance_data_hard_material = [2.7, 2.43, 2.17, 2.09, 2.01, 1.89, 1.86, 1.837, 1.825, 1.815, 1.8, 1.793, 1.78]  # y轴数据：Sensor Resistance

# # 空数组，用于存放在鞋垫上的数据（三角点）
# pressure_data_insole = [1.7,3,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,180,200,230,250]  # x轴数据：压强
# sensor_resistance_data_insole = [22,12,6,4,3.35,3,2.72,2.59,2.5,2.42,2.36,2.31,2.28,2.24,2.2,2.17,2.16,2.06,1.95,1.87,1.78]  # y轴数据：Sensor Resistance
#
# # 空数组，用于存放在鞋垫上的数据（三角点）
# pressure_data_insole = [300,305,316,330,350,379,397,424,437,473]  # x轴数据：压强
# sensor_resistance_data_insole = [1.965,1.87,1.825,1.78,1.725,1.69,1.67,1.65,1.65,1.65]  # y轴数据：Sensor Resistance