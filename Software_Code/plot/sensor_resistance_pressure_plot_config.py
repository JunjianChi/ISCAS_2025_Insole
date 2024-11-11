import matplotlib.pyplot as plt
import numpy as np

# 定义FigureConfig类
class FigureConfig:
    def __init__(self,
                 font='Arial',
                 font_bold=True,
                 font_size=22,
                 line_width=4,
                 axes_line_width=2,
                 grid=True,
                 legend=True,
                 legend_font_size=18,
                 tick_size=17,
                 marker_size=10,
                 grid_line_style='--',
                 grid_line_width=1,
                 title_size=26,
                 title_bold=True,
                 figsize=(10, 6)):
        self.font = font
        self.font_bold = font_bold
        self.font_size = font_size
        self.line_width = line_width
        self.axes_line_width = axes_line_width
        self.grid = grid
        self.legend = legend
        self.legend_font_size = legend_font_size
        self.tick_size = tick_size
        self.marker_size = marker_size
        self.grid_line_style = grid_line_style
        self.grid_line_width = grid_line_width
        self.title_size = title_size
        self.title_bold = title_bold
        self.figsize = figsize

    def apply(self):
        # 字体和样式设置
        rc_params = {
            'font.family': self.font,
            'font.size': self.font_size,
            'font.weight': 'bold' if self.font_bold else 'normal',
            'lines.linewidth': self.line_width,
            'axes.linewidth': self.axes_line_width,
            'axes.labelweight': 'bold' if self.font_bold else 'normal',
            'axes.labelsize': self.font_size,
            'axes.titlesize': self.title_size,
            'axes.titleweight': 'bold' if self.title_bold else 'normal',
            'xtick.labelsize': self.tick_size,
            'ytick.labelsize': self.tick_size,
            'xtick.major.width': self.line_width,
            'ytick.major.width': self.line_width,
            'xtick.major.size': self.marker_size,
            'ytick.major.size': self.marker_size,
            'grid.linestyle': self.grid_line_style,
            'grid.linewidth': self.grid_line_width,
            'axes.grid': self.grid,
            'legend.fontsize': self.legend_font_size,
            'legend.frameon': self.legend,
        }

        # 更新matplotlib的rc参数
        plt.rcParams.update(rc_params)

# 应用FigureConfig配置
config = FigureConfig()
config.apply()

# 原始数据
pressure = [2, 5.7, 11.9, 20, 29.9, 40, 60, 71.6, 80.3, 100, 111, 122, 131, 140, 150, 160]
R = [15.7, 9.05, 4.92, 2.79, 2.41, 2.15, 2.06, 2.02, 1.99, 1.9, 1.86, 1.81, 1.78, 1.76, 1.75, 1.75]

# 计算压力 (kPa)
area = np.pi * (0.01595) ** 2
pressure = np.array(pressure) / area * 0.001  # 转换为 kPa

# 定义水平线的y值
y_horizontal_line = R[-1]

# 硬性材料的数据（x点）
pressure_data_hard_material = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
sensor_resistance_data_hard_material = [2.7, 2.43, 2.17, 2.09, 2.01, 1.89, 1.86, 1.837, 1.825, 1.815, 1.8, 1.793, 1.78]

# 鞋垫上的数据（三角点）
pressure_data_insole1 = [1.7,3,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,180,200,230,250]
sensor_resistance_data_insole1 = [22,12,6,4,3.35,3,2.72,2.59,2.5,2.42,2.36,2.31,2.28,2.24,2.2,2.17,2.16,2.06,1.95,1.87,1.78]
pressure_data_insole2 = [300,305,316,330,350,379,397,424,437,473]
sensor_resistance_data_insole2 = [1.965,1.87,1.825,1.78,1.725,1.69,1.67,1.65,1.65,1.65]

# 合并鞋垫数据
pressure_data_insole = pressure_data_insole1 + pressure_data_insole2
sensor_resistance_data_insole = sensor_resistance_data_insole1 + sensor_resistance_data_insole2

# 创建图表
plt.figure(figsize=config.figsize)

# 0 kPa - 20 kPa 背景颜色
plt.axvspan(0, 25, color='lightgoldenrodyellow', alpha=0.8, label='0-25 kPa Region')

# 20 kPa - 200 kPa 背景颜色
plt.axvspan(25, 200, color='lightcyan', alpha=0.8, label='25-200 kPa Region')

# 绘制原始数据
plt.plot(pressure, R, markersize=config.marker_size, linestyle='-', color='mediumblue',marker='o', label='Sensor Resistance vs Pressure curve')

# 水平虚线，线宽3
plt.axhline(y=y_horizontal_line, linewidth=config.line_width, color='slategrey', linestyle='--',label= 'Sensor Resistance Saturation')



# # 绘制硬性材料数据（x点）
# plt.plot(pressure_data_hard_material, sensor_resistance_data_hard_material, marker='x', markersize=config.marker_size, linestyle='None', color='red', label='硬性材料数据')
#
# # 绘制鞋垫数据（三角点）
# plt.plot(pressure_data_insole, sensor_resistance_data_insole, marker='^', markersize=config.marker_size, linestyle='None', color='green', label='鞋垫数据')



# 设置标题和轴标签
# plt.title('Sensor Resistance vs Pressure', fontsize=config.title_size, fontweight='bold' if config.title_bold else 'normal')
plt.xlabel('Pressure (kPa)', fontsize=config.font_size, fontweight='bold' if config.font_bold else 'normal')
plt.ylabel('Sensor Resistance  (kΩ)', fontsize=config.font_size, fontweight='bold' if config.font_bold else 'normal')

# 设置y轴为对数尺度
plt.yscale('log')

# 定制坐标轴线条
ax = plt.gca()
ax.spines['top'].set_linewidth(config.axes_line_width)
ax.spines['right'].set_linewidth(config.axes_line_width)
ax.spines['left'].set_linewidth(config.axes_line_width)
ax.spines['bottom'].set_linewidth(config.axes_line_width)

# 设置刻度参数
plt.xticks(fontsize=config.tick_size)
plt.yticks(fontsize=config.tick_size)

# 添加网格
if config.grid:
    plt.grid(True, linestyle=config.grid_line_style, linewidth=config.grid_line_width)

# 显示图例
if config.legend:
    plt.legend(fontsize=config.legend_font_size, loc="upper right", frameon=config.legend)
# 保存图表为PNG格式，DPI为600
plt.savefig('sensor_resistance_vs_pressure.png', dpi=1000, bbox_inches='tight')
# 显示图表
plt.show()
