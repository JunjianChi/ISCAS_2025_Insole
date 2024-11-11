import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os

# 获取当前工作目录
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
file_path = os.path.join(parent_dir, 'dataset', 'walking', 'walk_5000f_cjj.csv')

# 读取CSV文件
data = pd.read_csv(file_path)

# 选择要可视化的帧，比如第1400帧
frame_index = 1000
frame_data = data.iloc[frame_index]

# 提取压力矩阵数据 (假设33x15的格式，按逗号分割)
left_matrix = np.array(frame_data['Matrix_0'].split(','), dtype=np.float32).reshape(33, 15)
right_matrix = np.array(frame_data['Matrix_1'].split(','), dtype=np.float32).reshape(33, 15)

# 提取关节点的三维信息，交换 Y 和 Z，并对 Y 取负数
joints = {
    'left_foot': [frame_data['left_foot_index_x'], frame_data['left_foot_index_z'], -frame_data['left_foot_index_y']],
    'right_foot': [frame_data['right_foot_index_x'], frame_data['right_foot_index_z'], -frame_data['right_foot_index_y']],
    'left_knee': [frame_data['left_knee_x'], frame_data['left_knee_z'], -frame_data['left_knee_y']],
    'right_knee': [frame_data['right_knee_x'], frame_data['right_knee_z'], -frame_data['right_knee_y']],
    'left_hip': [frame_data['left_hip_x'], frame_data['left_hip_z'], -frame_data['left_hip_y']],
    'right_hip': [frame_data['right_hip_x'], frame_data['right_hip_z'], -frame_data['right_hip_y']],
    'head': [frame_data['head_x'], frame_data['head_z'], -frame_data['head_y']],
    'left_shoulder': [frame_data['left_shoulder_x'], frame_data['left_shoulder_z'], -frame_data['left_shoulder_y']],
    'right_shoulder': [frame_data['right_shoulder_x'], frame_data['right_shoulder_z'], -frame_data['right_shoulder_y']]
}

# 创建绘图
fig = plt.figure(figsize=(12, 6))

# 1. 左脚压力矩阵热力图
ax1 = fig.add_subplot(1, 3, 1)
sns.heatmap(left_matrix, cmap="YlGnBu", ax=ax1)
ax1.set_title('Left Foot Pressure Matrix')

# 2. 右脚压力矩阵热力图
ax2 = fig.add_subplot(1, 3, 2)
sns.heatmap(right_matrix, cmap="YlGnBu", ax=ax2)
ax2.set_title('Right Foot Pressure Matrix')

# 3. 三维关节点图
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.set_xlabel('X (Horizontal)')
ax3.set_ylabel('Z (Depth)')
ax3.set_zlabel('Y (Vertical)')
ax3.set_title('3D Joint Positions (Y and Z Swapped)')

# 提取关键点的坐标并绘制 (交换 Y 和 Z 坐标并对 Y 取负数)
for joint, coord in joints.items():
    if joint == 'head':
        # 将头部节点的大小设为更大
        ax3.scatter(coord[0], coord[1], coord[2], label=joint, s=100)  # 头部大小设置为200
    else:
        ax3.scatter(coord[0], coord[1], coord[2], label=joint)

    ax3.text(coord[0], coord[1], coord[2], joint, fontsize=9)

# 定义要连接的关节点对
connections = [
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_foot'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_foot'),
    ('left_hip', 'right_hip'),
    ('left_shoulder', 'right_shoulder'),
    ('left_hip', 'left_shoulder'),
    ('right_hip', 'right_shoulder')

]

# 画出关节点之间的连线 (交换 Y 和 Z 坐标并对 Y 取负数)
for joint1, joint2 in connections:
    x_vals = [joints[joint1][0], joints[joint2][0]]  # X -> X
    z_vals = [joints[joint1][2], joints[joint2][2]]  # Y -> Z (交换 Y 和 Z)
    y_vals = [joints[joint1][1], joints[joint2][1]]  # Z -> Y
    ax3.plot(x_vals, y_vals, z_vals, 'k-')

# 调整视角，使得 X 为水平轴，Y 为垂直轴，Z 为深度轴
ax3.view_init(elev=15., azim=210)  # elev 设置仰角，azim 设置水平旋转角度

# ax3.legend()

plt.tight_layout()
plt.show()
