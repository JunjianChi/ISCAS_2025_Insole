import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# Read the CSV file
# 获取当前工作目录
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
file_path = os.path.join(parent_dir, 'dataset', 'walking', 'walking_ritchie.csv')

# 读取CSV文件
data = pd.read_csv(file_path)

# Convert Matrix_1 string into a list of integers for both left and right
data['Matrix_1_list'] = data['Matrix_0'].apply(lambda x: [int(i) for i in x.split(',')])
data['Matrix_2_list'] = data['Matrix_1'].apply(lambda x: [int(i) for i in x.split(',')])  # Assuming right foot Matrix_2

# Extract the value at position [10][11] from Matrix_1 for left foot, assuming it's an 11x11 matrix
matrix_1_values = [matrix[110] if len(matrix) > 121 else None for matrix in data['Matrix_1_list']]

# Extract the value at position [10][11] from Matrix_2 for right foot, assuming it's an 11x11 matrix
matrix_2_values = [matrix[110] if len(matrix) > 121 else None for matrix in data['Matrix_2_list']]

# Calculate the distance between the left and right knee in 3D space
knee_distance = np.sqrt(
    (data['right_knee_x'] - data['left_knee_x'])**2 +
    (data['right_knee_y'] - data['left_knee_y'])**2 +
    (data['right_knee_z'] - data['left_knee_z'])**2
)

# Calculate the distance between left and right foot index in 3D space
foot_index_distance = np.sqrt(
    (data['right_foot_index_x'] - data['left_foot_index_x'])**2 +
    (data['right_foot_index_y'] - data['left_foot_index_y'])**2 +
    (data['right_foot_index_z'] - data['left_foot_index_z'])**2
)

# Define a normalization function
def normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series))

# Apply normalization
normalized_foot_z = normalize(data['right_foot_index_z'])
normalized_left_foot_z = normalize(data['left_foot_index_z'])  # Normalize left foot z-axis
normalized_matrix_1 = normalize(matrix_1_values)  # Normalize left foot matrix
normalized_matrix_2 = normalize(matrix_2_values)  # Normalize right foot matrix
normalized_knee_distance = normalize(knee_distance)
normalized_foot_index_distance = normalize(foot_index_distance)

# Define a moving average function for smoothing
def moving_average(series, window_size=20):
    return series.rolling(window=window_size, min_periods=1).mean()

# Apply moving average to smooth the series
smoothed_foot_z = moving_average(pd.Series(normalized_foot_z))
smoothed_left_foot_z = moving_average(pd.Series(normalized_left_foot_z))  # Smoothed left foot z-axis
smoothed_matrix_1 = moving_average(pd.Series(normalized_matrix_1))  # Smoothed left foot matrix
smoothed_matrix_2 = moving_average(pd.Series(normalized_matrix_2))  # Smoothed right foot matrix
smoothed_knee_distance = moving_average(pd.Series(normalized_knee_distance))
smoothed_foot_index_distance = moving_average(pd.Series(normalized_foot_index_distance))

# Plot the normalized and smoothed data
plt.figure(figsize=(10, 6))

# Plot smoothed left foot z-axis
plt.plot(data['Frame'], smoothed_left_foot_z, label='Smoothed Left foot z-axis coordinate', color='blue', linewidth=2)

# Plot smoothed right foot z-axis
plt.plot(data['Frame'], smoothed_foot_z, label='Smoothed Right foot z-axis coordinate', color='red', linewidth=2)

# Plot smoothed Matrix_0 [10][11] value for left foot
plt.plot(data['Frame'], smoothed_matrix_1, label='Smoothed Matrix_1 [10][11] Left Foot', linestyle='--', color='green', linewidth=2)

# Plot smoothed Matrix_2 [10][11] value for right foot
plt.plot(data['Frame'], smoothed_matrix_2, label='Smoothed Matrix_2 [10][11] Right Foot',  linestyle='--', color='orange', linewidth=2)

# Plot smoothed distance between left and right knee in 3D space
plt.plot(data['Frame'], smoothed_knee_distance, label='Smoothed 3D distance between left and right knee', linestyle='-.', color='purple', linewidth=2)

# Plot smoothed distance between left and right foot index in 3D space
plt.plot(data['Frame'], smoothed_foot_index_distance, label='Smoothed Distance between left and right foot index', linestyle='-.', color='brown', linewidth=2)

# Add labels and title
plt.xlabel('Frame')
plt.ylabel('Normalized and Smoothed Value')
plt.title('Smoothed and Normalized Foot Z-axis, Matrix_1 and Matrix_2 [10][11], Knee, and Foot Index Distances over Frames')
plt.grid(True)
plt.legend()

# Show plot
plt.show()
