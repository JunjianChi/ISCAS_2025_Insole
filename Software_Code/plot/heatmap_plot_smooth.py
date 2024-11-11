import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom, shift
from matplotlib.colors import LinearSegmentedColormap
import os
import cv2

# 定义常量
FRAME_START = 0
FRAME_END = 1000  # 包含帧1000
ZOOM_FACTOR_SMOOTH = 1.2  # 平滑和矩阵放大因子
ZOOM_FACTOR_MASK = 1.2    # 模式放大因子，与ZOOM_FACTOR_SMOOTH一致
SIGMA = 1.0
V_MAX = 1800  # 根据需要调整

def generate_right_foot_pattern(left_pattern):
    """
    通过左右对称生成右脚鞋垫模式。

    参数：
    - left_pattern: 左脚鞋垫的压力分布（二维列表）

    返回：
    - right_pattern: 右脚鞋垫的压力分布（二维列表）
    """
    right_pattern = [row[::-1] for row in left_pattern]  # 左右对称
    return right_pattern

def generate_mask_from_image(image_path, target_rows=33, target_cols=15, lower_green=np.array([40, 40, 40]), upper_green=np.array([70, 255, 255])):
    """
    从图像生成鞋垫模式。

    参数：
    - image_path: 图像文件的路径
    - target_rows: 模式的行数
    - target_cols: 模式的列数
    - lower_green: HSV颜色空间中绿色的下界
    - upper_green: HSV颜色空间中绿色的上界

    返回：
    - mask_resized: 一个二维列表，表示鞋垫的压力分布（1 表示有效压力点，0 表示无压力点）
    """
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")

    # 旋转图像90度顺时针
    image_rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    # 转换为HSV颜色空间
    hsv = cv2.cvtColor(image_rotated, cv2.COLOR_BGR2HSV)

    # 创建绿色掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("未检测到绿色边界。请检查图像中的绿色标记是否正确。")

    # 创建空白掩码
    mask_filled = np.zeros_like(mask)

    # 填充所有检测到的轮廓
    cv2.drawContours(mask_filled, contours, -1, color=255, thickness=cv2.FILLED)

    # 二值化处理
    _, binary_mask = cv2.threshold(mask_filled, 127, 255, cv2.THRESH_BINARY)

    # 调整掩码大小
    mask_resized = cv2.resize(binary_mask, (target_cols, target_rows), interpolation=cv2.INTER_NEAREST)

    # 转换为1和0
    mask_binary = (mask_resized > 0).astype(int)

    return mask_binary.tolist()

def smooth_and_interpolate_matrix(matrix_str, sigma=SIGMA, zoom_factor=ZOOM_FACTOR_SMOOTH):
    """
    平滑和插值处理矩阵数据。

    参数：
    - matrix_str: CSV中的矩阵字符串
    - sigma: 高斯模糊的标准差
    - zoom_factor: 插值放大因子

    返回：
    - smooth_matrix: 处理后的平滑矩阵
    """
    try:
        matrix = np.array(matrix_str.split(','), dtype=np.float32).reshape(33, 15)
    except ValueError:
        raise ValueError("Matrix data does not have the expected shape of 33x15.")

    # 插值放大使矩阵更平滑
    interpolated_matrix = zoom(matrix, zoom_factor, order=1)  # 双线性插值
    # 应用高斯模糊进行平滑处理
    smooth_matrix = gaussian_filter(interpolated_matrix, sigma=sigma)
    print(f"Smoothed matrix shape: {smooth_matrix.shape}")  # 调试输出
    return smooth_matrix

def extract_pressure_points(matrix, pattern, zoom_factor=ZOOM_FACTOR_MASK, x_shift=0, y_shift=0):
    """
    根据模式提取有效压力点，并通过双线性插值获得更平滑的边界。

    参数：
    - matrix: 平滑后的压力矩阵
    - pattern: 鞋垫的压力模式（二维列表）
    - zoom_factor: 模式放大因子
    - x_shift: 水平位移（正值向右，负值向左）
    - y_shift: 垂直位移（正值向下，负值向上）

    返回：
    - pressure_points: 应用掩码后的压力矩阵（无效点为NaN）
    """
    mask = np.array(pattern, dtype=float)  # 使用浮点数以支持插值
    try:
        # 使用双线性插值（order=1）放大掩码
        interpolated_mask = zoom(mask, zoom_factor, order=1)
    except ValueError:
        raise ValueError("Zoom factor results in a mask size that does not match the matrix size.")

    # 应用平移到掩码
    if x_shift != 0 or y_shift != 0:
        interpolated_mask = shift(interpolated_mask, shift=(y_shift, x_shift), mode='nearest')

    # 通过阈值将插值后的掩码转换为二值形式
    binary_mask = interpolated_mask > 0.5  # 阈值可以根据需要调整

    # 确保掩码和矩阵形状一致
    if binary_mask.shape != matrix.shape:
        raise ValueError(f"Mask shape {binary_mask.shape} does not match matrix shape {matrix.shape}.")

    pressure_points = np.where(binary_mask, matrix, np.nan)
    return pressure_points

# 自定义颜色映射
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'green', 'yellow', 'red'])

def plot_pressure_heatmap(data_matrix_0, data_matrix_1, left_pattern, right_pattern, frame_index, vmax=None,
                         x_shift_left=0, y_shift_left=0, x_shift_right=0, y_shift_right=0, gap_size=2,
                         save_path=None):
    """
    绘制压力热力图并保存为PNG文件。

    参数：
    - data_matrix_0: 左脚的压力数据（Series）
    - data_matrix_1: 右脚的压力数据（Series）
    - left_pattern: 左脚鞋垫的压力模式（二维列表）
    - right_pattern: 右脚鞋垫的压力模式（二维列表）
    - frame_index: 要绘制的帧索引
    - vmax: 色条的最大值
    - x_shift_left: 左脚的水平位移
    - y_shift_left: 左脚的垂直位移
    - x_shift_right: 右脚的水平位移
    - y_shift_right: 右脚的垂直位移
    - gap_size: 左右脚之间的间隙大小（以列为单位）
    - save_path: 保存PNG文件的路径。如果为None，则不保存。
    """
    if frame_index < 0 or frame_index >= len(data_matrix_0):
        raise IndexError(f"frame_index {frame_index} is out of bounds for data with {len(data_matrix_0)} frames.")
    print(left_pattern)
    # 平滑和插值处理矩阵
    matrix_0 = smooth_and_interpolate_matrix(data_matrix_0[frame_index], sigma=SIGMA, zoom_factor=ZOOM_FACTOR_SMOOTH)
    matrix_1 = smooth_and_interpolate_matrix(data_matrix_1[frame_index], sigma=SIGMA, zoom_factor=ZOOM_FACTOR_SMOOTH)

    # 根据模式提取有效压力点，并应用平移
    left_pressure = extract_pressure_points(matrix_0, left_pattern, zoom_factor=ZOOM_FACTOR_MASK,
                                           x_shift=x_shift_left, y_shift=y_shift_left)
    right_pressure = extract_pressure_points(matrix_1, right_pattern, zoom_factor=ZOOM_FACTOR_MASK,
                                            x_shift=x_shift_right, y_shift=y_shift_right)

    # 创建间隙
    gap = np.full((left_pressure.shape[0], gap_size), np.nan)

    # 拼接左脚、间隙和右脚的压力矩阵
    combined_pressure = np.hstack((left_pressure, gap, right_pressure))

    print(f"Combined pressure shape: {combined_pressure.shape}")  # 调试输出

    # 创建图形
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(combined_pressure, cmap=custom_cmap, cbar=True, vmax=vmax, vmin=0,
                     cbar_kws={'label': 'Pressure (kPa)'}, square=True)

    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 50, 100, 150, 200])
    cbar.set_label('Pressure (kPa)', fontsize=30, fontweight='bold')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=30, fontweight='bold')

    # 移除坐标轴的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')

    # 提高图像的DPI以提高清晰度
    plt.gcf().set_dpi(1000)

    # 设置标题（如果需要，可以取消注释）
    # plt.title(f'Combined Left and Right Foot Pressure (Frame {FRAME_START + frame_index})', fontsize=14, fontweight='bold')

    # 调整布局
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"图像已保存到 {save_path}")
    else:
        plt.show()
    plt.show()
    # 关闭图形以释放内存
    plt.close()

def main():
    # 确定文件路径
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    file_path = os.path.join(parent_dir, 'dataset', 'standing', 'plotting.csv')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV文件未找到，路径为 {file_path}")

    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 验证所需列是否存在
    required_columns = {'Matrix_0', 'Matrix_1'}
    if not required_columns.issubset(data.columns):
        missing = required_columns - set(data.columns)
        raise ValueError(f"CSV文件缺少列: {missing}")

    # 提取指定帧的数据
    if FRAME_END >= len(data):
        raise ValueError(f"FRAME_END {FRAME_END} 超过了数据集的总帧数 {len(data)}。")

    data = data.iloc[FRAME_START:FRAME_END + 1].reset_index(drop=True)

    data_matrix_0 = data['Matrix_0']
    data_matrix_1 = data['Matrix_1']

    # 生成左脚和右脚的图像化模式
    # 请将 'left_insole.png' 替换为您的实际图像路径
    left_insole_image_path = os.path.join(parent_dir, 'dataset', 'standing', 'left_insole.png')

    # 检查图像文件是否存在
    if not os.path.exists(left_insole_image_path):
        raise FileNotFoundError(f"左脚鞋垫图像未找到，路径为 {left_insole_image_path}")

    # 生成左脚模式
    left_pattern = generate_mask_from_image(
        left_insole_image_path,
        target_rows=33,
        target_cols=15,
        lower_green=np.array([40, 40, 40]),  # 根据需要调整
        upper_green=np.array([70, 255, 255])  # 根据需要调整
    )

    # 修改左脚模式（根据需要调整）
    left_pattern = np.array(left_pattern)
    left_pattern[:, -1] = 0
    left_pattern[-9:-1, 12] = 0
    left_pattern[-1, :] = 0
    left_pattern[-2, 11] = 0

    # 生成右脚模式
    right_pattern = generate_right_foot_pattern(left_pattern)

    # 指定要绘制的帧
    desired_frame = 124 # 您希望绘制的实际帧编号

    # 计算相对于切片数据的帧索引
    frame_to_plot = desired_frame - FRAME_START

    if frame_to_plot < 0 or frame_to_plot >= len(data_matrix_0):
        print(f"帧索引 {desired_frame} 超出范围。可用的帧范围是从 {FRAME_START} 到 {FRAME_END}。")
        return

    # 定义掩码的平移量（根据需要调整）
    # 左脚的平移
    x_shift_left = -1  # 正值向右，负值向左
    y_shift_left = 0  # 正值向下，负值向上

    # 右脚的平移
    x_shift_right = -1  # 正值向右，负值向左
    y_shift_right = 0  # 正值向下，负值向上

    # 定义左右脚之间的间隙大小
    gap_size = 2  # 可以根据需要调整

    # 定义保存PNG文件的目录
    output_dir = os.path.join(parent_dir, 'output_heatmaps')
    os.makedirs(output_dir, exist_ok=True)

    # 定义PNG文件的名称
    save_filename = f'pressure_frame_{desired_frame}.png'
    save_path = os.path.join(output_dir, save_filename)

    # 绘制并保存压力热力图
    plot_pressure_heatmap(
        data_matrix_0=data_matrix_0,
        data_matrix_1=data_matrix_1,
        left_pattern=left_pattern,
        right_pattern=right_pattern,
        frame_index=frame_to_plot,
        vmax=V_MAX,
        x_shift_left=x_shift_left,
        y_shift_left=y_shift_left,
        x_shift_right=x_shift_right,
        y_shift_right=y_shift_right,
        gap_size=gap_size,
        save_path=save_path  # 传递保存路径
    )

if __name__ == "__main__":
    main()
