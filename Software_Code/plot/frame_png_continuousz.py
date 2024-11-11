import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# 创建文件夹
output_dir = "output_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 获取当前工作目录
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
file_path = os.path.join(parent_dir, 'dataset', 'walking', 'walking_cjj.csv')

# 读取CSV文件
data = pd.read_csv(file_path)

# 定义颜色映射，确保每个关节点在每帧中颜色一致
color_map = {
    'left_foot': 'deepskyblue',
    'right_foot': 'steelblue',
    'left_knee': 'lightseagreen',
    'right_knee': 'darkcyan',
    'left_hip': 'teal',
    'right_hip': 'slategray',
    'head': 'dimgray',
    'left_shoulder': 'lightsteelblue',
    'right_shoulder': 'dodgerblue'
}

# 选择要可视化的帧数
selected_frames = [1606, 1610, 1616, 1622, 1632]

# 创建一个新的 3D 图
fig3 = plt.figure(figsize=(8, 8))  # 增大图像尺寸
ax4 = fig3.add_subplot(111, projection='3d')

# 设置坐标轴标签和标题
ax4.set_xlabel('X (Horizontal)', fontsize=10, fontweight='bold', labelpad=10)
ax4.set_ylabel('    Z (Depth)', fontsize=10, fontweight='bold', labelpad=20)
ax4.set_zlabel('Y (Vertical)', fontsize=10, fontweight='bold', labelpad=10)
ax4.set_title('3D Joint Positions', fontsize=16, fontweight='bold', pad=1)
ax4.set_box_aspect([4, 10, 4])  # X:Y:Z 的比例，这里Z轴更长

ax4.set_xticks(np.linspace(-1, 1, 10))  # 示例：减少X轴刻度数量
ax4.set_yticks(np.linspace(0, 1000, 30))  # 示例：减少Y轴刻度数量
ax4.set_zticks(np.linspace(-1, 1, 10))  # 示例：减少Z轴刻度数量

# 移除背景和网格线，使背景透明
ax4.set_facecolor((0, 0, 0, 0))  # 设置轴的背景为透明

# 移除轴面
ax4.xaxis.pane.fill = False
ax4.yaxis.pane.fill = False
ax4.zaxis.pane.fill = False

# 移除轴的边框线
ax4.xaxis.pane.set_edgecolor((1, 1, 1, 0))
ax4.yaxis.pane.set_edgecolor((1, 1, 1, 0))
ax4.zaxis.pane.set_edgecolor((1, 1, 1, 0))

# 关闭网格线
ax4.grid(False)

# 定义要连接的关节点对，添加头部和肩部的连接
connections = [
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_foot'),
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_foot'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_shoulder'),
    ('right_hip', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'head'),
    ('right_shoulder', 'head'),
]

# 设置Z轴拉长的缩放因子
z_scale_factor = 100  # 你可以根据需要调整这个值
depth_increment = 100  # 每一帧在 Z 轴上额外增加的深度

# 逐帧绘制关节点
for frame_idx, frame_index in enumerate(selected_frames):
    frame_data = data.iloc[frame_index]

    # 计算新的 Z 轴深度
    z_offset = frame_idx * depth_increment  # 随着帧数增加，Z 轴值递增

    # 提取每个关节点的坐标，拉长Z轴，添加头部和肩部
    joints = {
        'left_foot': [frame_data['left_foot_index_x'], frame_data['left_foot_index_z'] * z_scale_factor + z_offset,
                      -frame_data['left_foot_index_y']],
        'right_foot': [frame_data['right_foot_index_x'], frame_data['right_foot_index_z'] * z_scale_factor + z_offset,
                       -frame_data['right_foot_index_y']],
        'left_knee': [frame_data['left_knee_x'], frame_data['left_knee_z'] * z_scale_factor + z_offset,
                      -frame_data['left_knee_y']],
        'right_knee': [frame_data['right_knee_x'], frame_data['right_knee_z'] * z_scale_factor + z_offset,
                       -frame_data['right_knee_y']],
        'left_hip': [frame_data['left_hip_x'], frame_data['left_hip_z'] * z_scale_factor + z_offset,
                     -frame_data['left_hip_y']],
        'right_hip': [frame_data['right_hip_x'], frame_data['right_hip_z'] * z_scale_factor + z_offset,
                      -frame_data['right_hip_y']],
        'left_shoulder': [frame_data['left_shoulder_x'], frame_data['left_shoulder_z'] * z_scale_factor + z_offset,
                          -frame_data['left_shoulder_y']],
        'right_shoulder': [frame_data['right_shoulder_x'], frame_data['right_shoulder_z'] * z_scale_factor + z_offset,
                           -frame_data['right_shoulder_y']],
        'head': [frame_data['head_x'], frame_data['head_z'] * z_scale_factor + z_offset,
                 -frame_data['head_y']],
    }

    # 绘制关节点并使用颜色映射
    for joint, coord in joints.items():
        ax4.scatter(coord[0], coord[1], coord[2], label=f'{joint}' if frame_idx == 0 else "", color=color_map.get(joint, 'black'),
                    s=200)

    # 画出关节点之间的连线
    for joint1, joint2 in connections:
        x_vals = [joints[joint1][0], joints[joint2][0]]  # X -> X
        z_vals = [joints[joint1][2], joints[joint2][2]]  # Y -> Z (交换 Y 和 Z)
        y_vals = [joints[joint1][1], joints[joint2][1]]  # Z -> Y
        ax4.plot(x_vals, y_vals, z_vals, 'k-', linewidth=2)

# 调整视角，使得 X 为水平轴，Y 为垂直轴，Z 为深度轴
ax4.view_init(elev=10, azim=220)

# 保存3D关节点图，DPI 300 提高清晰度，设置背景透明
joint_image_path = os.path.join(output_dir, "3d_joint_positions_multiple_frames_with_labels.png")
plt.savefig(joint_image_path, dpi=1000, transparent=True)

plt.show()

print(f"Images saved in folder: {output_dir}")
