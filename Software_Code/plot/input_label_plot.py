import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
file_path = os.path.join(parent_dir, 'dataset', 'walking', 'walking_cjj.csv')

# 读取CSV文件
data = pd.read_csv(file_path)



# Convert Matrix_1 string into a list of integers for both left and right
data['Matrix_1_list'] = data['Matrix_0'].apply(lambda x: [int(i) for i in x.split(',')])
data['Matrix_2_list'] = data['Matrix_1'].apply(lambda x: [int(i) for i in x.split(',')])  # Assuming right foot Matrix_2

# Extract the value at position [10][11] from Matrix_1 for left foot
matrix_1_values = [matrix[110] if len(matrix) > 121 else None for matrix in data['Matrix_1_list']]

# Extract the value at position [10][11] from Matrix_2 for right foot
matrix_2_values = [matrix[110] if len(matrix) > 121 else None for matrix in data['Matrix_2_list']]

# 计算髋关节的中点
left_hip = np.array([data['left_hip_x'], data['left_hip_y'], data['left_hip_z']]).T
right_hip = np.array([data['right_hip_x'], data['right_hip_y'], data['right_hip_z']]).T
mid_hip = (left_hip + right_hip) / 2  # 髋关节中点

# 计算膝盖和脚相对于髋关节中点的相对位置（label）
relative_left_knee = np.array([data['left_knee_x'], data['left_knee_y'], data['left_knee_z']]).T - mid_hip
relative_right_knee = np.array([data['right_knee_x'], data['right_knee_y'], data['right_knee_z']]).T - mid_hip
relative_left_foot = np.array([data['left_foot_index_x'], data['left_foot_index_y'], data['left_foot_index_z']]).T - mid_hip
relative_right_foot = np.array([data['right_foot_index_x'], data['right_foot_index_y'], data['right_foot_index_z']]).T - mid_hip

# 定义归一化函数
def normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series))

# Apply normalization to inputs and labels
normalized_matrix_1 = normalize(matrix_1_values)  # Normalize left foot matrix
normalized_matrix_2 = normalize(matrix_2_values)  # Normalize right foot matrix
normalized_left_foot_z = normalize(data['left_foot_index_z'])  # Normalize left foot z-axis
normalized_right_foot_z = normalize(data['right_foot_index_z'])  # Normalize right foot z-axis

# Normalize the relative coordinates (labels)
normalized_relative_left_knee = normalize(relative_left_knee)
normalized_relative_right_knee = normalize(relative_right_knee)
normalized_relative_left_foot = normalize(relative_left_foot)
normalized_relative_right_foot = normalize(relative_right_foot)

# Apply moving average for smoothing
def moving_average(series, window_size=20):
    return series.rolling(window=window_size, min_periods=1).mean()

# Smoothing inputs
smoothed_matrix_1 = moving_average(pd.Series(normalized_matrix_1))
smoothed_matrix_2 = moving_average(pd.Series(normalized_matrix_2))
smoothed_left_foot_z = moving_average(pd.Series(normalized_left_foot_z))
smoothed_right_foot_z = moving_average(pd.Series(normalized_right_foot_z))

# Smoothing labels (relative positions)
smoothed_relative_left_knee = moving_average(pd.DataFrame(normalized_relative_left_knee))
smoothed_relative_right_knee = moving_average(pd.DataFrame(normalized_relative_right_knee))
smoothed_relative_left_foot = moving_average(pd.DataFrame(normalized_relative_left_foot))
smoothed_relative_right_foot = moving_average(pd.DataFrame(normalized_relative_right_foot))

# 绘制 Input 数据
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(data['Frame'], smoothed_left_foot_z, label='Smoothed Left foot z-axis coordinate', color='blue', linewidth=2)
plt.plot(data['Frame'], smoothed_right_foot_z, label='Smoothed Right foot z-axis coordinate', color='red', linewidth=2)
plt.plot(data['Frame'], smoothed_matrix_1, label='Smoothed Matrix_1 [10][11] Left Foot', linestyle='--', color='green', linewidth=2)
plt.plot(data['Frame'], smoothed_matrix_2, label='Smoothed Matrix_2 [10][11] Right Foot', linestyle='--', color='orange', linewidth=2)
plt.title('Input Data: Foot Z-axis and Pressure Matrix Values')
plt.xlabel('Frame')
plt.ylabel('Normalized and Smoothed Value')
plt.legend()
plt.grid(True)

# 绘制 Label 数据（相对关节三维坐标）
plt.subplot(1, 2, 2)
plt.plot(data['Frame'], smoothed_relative_left_knee[0], label='Smoothed Left Knee X', color='purple', linewidth=2)
plt.plot(data['Frame'], smoothed_relative_left_knee[1], label='Smoothed Left Knee Y', linestyle='--', color='purple', linewidth=2)
plt.plot(data['Frame'], smoothed_relative_left_knee[2], label='Smoothed Left Knee Z', linestyle='-.', color='purple', linewidth=2)

plt.plot(data['Frame'], smoothed_relative_right_knee[0], label='Smoothed Right Knee X', color='brown', linewidth=2)
plt.plot(data['Frame'], smoothed_relative_right_knee[1], label='Smoothed Right Knee Y', linestyle='--', color='brown', linewidth=2)
plt.plot(data['Frame'], smoothed_relative_right_knee[2], label='Smoothed Right Knee Z', linestyle='-.', color='brown', linewidth=2)

plt.title('Label Data: Smoothed Relative Knee 3D Coordinates')
plt.xlabel('Frame')
plt.ylabel('Normalized and Smoothed Value')
plt.legend()
plt.grid(True)

# Show the plots
plt.tight_layout()
plt.show()
