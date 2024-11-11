import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 获取当前工作目录
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
file_path = os.path.join(parent_dir, 'dataset', 'walking', 'walk_5000f_cjj.csv')

# 读取CSV文件
data = pd.read_csv(file_path)

# Convert Matrix_1 string into a list of integers for both left and right
data['Matrix_1_list'] = data['Matrix_0'].apply(lambda x: [int(i) for i in x.split(',')])
data['Matrix_2_list'] = data['Matrix_1'].apply(lambda x: [int(i) for i in x.split(',')])  # Assuming right foot Matrix_2

# Extract the value at position [10][11] from Matrix_1 for left foot, assuming it's a 33x15 matrix
matrix_1_values = [matrix[110] if len(matrix) >= 110 else None for matrix in data['Matrix_1_list']]

# Extract the value at position [10][11] from Matrix_2 for right foot, assuming it's a 33x15 matrix
matrix_2_values = [matrix[110] if len(matrix) >= 110 else None for matrix in data['Matrix_2_list']]

# Calculate the distance between left hip and left knee in 3D space
left_hip_knee_distance = np.sqrt(
    (data['left_hip_x'] - data['left_knee_x'])**2 +
    (data['left_hip_y'] - data['left_knee_y'])**2 +
    (data['left_hip_z'] - data['left_knee_z'])**2
)

# Calculate the distance between left knee and left foot in 3D space
left_knee_foot_distance = np.sqrt(
    (data['left_knee_x'] - data['left_foot_index_x'])**2 +
    (data['left_knee_y'] - data['left_foot_index_y'])**2 +
    (data['left_knee_z'] - data['left_foot_index_z'])**2
)

# Calculate the distance between right hip and right knee in 3D space
right_hip_knee_distance = np.sqrt(
    (data['right_hip_x'] - data['right_knee_x'])**2 +
    (data['right_hip_y'] - data['right_knee_y'])**2 +
    (data['right_hip_z'] - data['right_knee_z'])**2
)

# Calculate the distance between right knee and right foot in 3D space
right_knee_foot_distance = np.sqrt(
    (data['right_knee_x'] - data['right_foot_index_x'])**2 +
    (data['right_knee_y'] - data['right_foot_index_y'])**2 +
    (data['right_knee_z'] - data['right_foot_index_z'])**2
)

# Define a normalization function
def normalize(series):
    return (series - np.min(series)) / (np.max(series) - np.min(series))

# Apply normalization
normalized_foot_z = normalize(data['right_foot_index_z'])
normalized_left_foot_z = normalize(data['left_foot_index_z'])  # Normalize left foot z-axis
normalized_matrix_1 = normalize(matrix_1_values)  # Normalize left foot matrix
normalized_matrix_2 = normalize(matrix_2_values)  # Normalize right foot matrix

# Define a function to calculate the angle between two vectors
def calculate_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_theta)
    return np.degrees(angle)  # Convert from radians to degrees

# Calculate vectors for left side
left_hip_knee_vector = np.array([data['left_knee_x'] - data['left_hip_x'],
                                 data['left_knee_y'] - data['left_hip_y'],
                                 data['left_knee_z'] - data['left_hip_z']]).T

left_knee_foot_vector = np.array([data['left_foot_index_x'] - data['left_knee_x'],
                                  data['left_foot_index_y'] - data['left_knee_y'],
                                  data['left_foot_index_z'] - data['left_knee_z']]).T

# Calculate the angle for each frame for the left knee
left_knee_angles = [calculate_angle(left_hip_knee_vector[i], left_knee_foot_vector[i]) for i in range(len(data))]

# Calculate vectors for right side
right_hip_knee_vector = np.array([data['right_knee_x'] - data['right_hip_x'],
                                  data['right_knee_y'] - data['right_hip_y'],
                                  data['right_knee_z'] - data['right_hip_z']]).T

right_knee_foot_vector = np.array([data['right_foot_index_x'] - data['right_knee_x'],
                                   data['right_foot_index_y'] - data['right_knee_y'],
                                   data['right_foot_index_z'] - data['right_knee_z']]).T

# Calculate the angle for each frame for the right knee
right_knee_angles = [calculate_angle(right_hip_knee_vector[i], right_knee_foot_vector[i]) for i in range(len(data))]

# Plot the normalized and smoothed data
plt.figure(figsize=(10, 6))

# Plot left knee angle
plt.plot(data['Frame'], left_knee_angles, label='Left Knee Angle (Degrees)', color='blue', linewidth=2)

# Plot right knee angle
plt.plot(data['Frame'], right_knee_angles, label='Right Knee Angle (Degrees)', color='red', linewidth=2)

# Plot smoothed Matrix_0 [10][11] value for left foot
plt.plot(data['Frame'], normalized_matrix_1, label='Matrix_1 [10][11] Left Foot', linestyle='--', color='green', linewidth=2)

# Plot smoothed Matrix_2 [10][11] value for right foot
plt.plot(data['Frame'], normalized_matrix_2, label='Matrix_2 [10][11] Right Foot',  linestyle='--', color='orange', linewidth=2)

# Add labels and title
plt.xlabel('Frame')
plt.ylabel('Knee Angle (Degrees) / Pressure Value')
plt.title('Knee Angles and Pressure Matrices (Left and Right) over Frames')
plt.grid(True)
plt.legend()

# Show plot
plt.show()
