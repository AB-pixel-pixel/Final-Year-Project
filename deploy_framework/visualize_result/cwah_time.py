import os
import pandas as pd

# 定义路径
root_dir = '/media/airs/BIN/cwah_ex/week2/gpt-4o-mini_20250305_142923'
scene_dirs = [d for d in os.listdir(root_dir) if d.startswith('scene')]

total_time = 0
total_scenes = 0

# 遍历每个 scene 文件夹
for scene in scene_dirs:
    csv_path = os.path.join(root_dir, scene, 'stage_time_consumption.csv')
    
    # 检查文件是否存在
    if os.path.isfile(csv_path):
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)
        
        # 累加 Accumulated Inference Time
        if 'Accumulated Inference Time' in df.columns:
            total_time += df['Accumulated Inference Time'].sum()
            total_scenes += 1

# 计算平均用时
average_time = total_time / total_scenes if total_scenes > 0 else 0

# 输出结果
print(f"总共用时: {total_time:.4f}秒")
print(f"场景数量: {total_scenes}")
print(f"平均用时: {average_time:.4f}秒")