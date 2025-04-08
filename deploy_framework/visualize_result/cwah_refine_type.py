# import os
# import json

# # 存储所有场景的调用次数
# total_counts = {}

# # 遍历所有 scene 文件夹
# scene_dir = '/media/airs/BIN/cwah_ex/week2/gpt-4o-mini_20250221_143816/'
# for scene in os.listdir(scene_dir):
#     scene_path = os.path.join(scene_dir, scene)
    
#     # 确保是目录并且名称以 "scene" 开头
#     if os.path.isdir(scene_path) and scene.startswith('scene'):
#         json_file_path = os.path.join(scene_path, 'refine_type_statistical_results.json')
        
#         # 读取 JSON 文件
#         if os.path.exists(json_file_path):
#             with open(json_file_path, 'r') as file:
#                 data = json.load(file)
                
#                 # 更新总次数
#                 for key, value in data.items():
#                     if key in total_counts:
#                         total_counts[key] += value
#                     else:
#                         total_counts[key] = value

# # 统计总调用次数
# total_calls = sum(total_counts.values())

# # 计算每个步骤的频率
# frequency = {key: value / total_calls for key, value in total_counts.items()}

# # 输出结果
# print("调用次数:", total_counts)
# print("总调用次数:", total_calls)
# print("调用频率:", frequency)

import os
import json
import pandas as pd

# 存储所有场景的调用次数
total_counts = {}

# 遍历所有 scene 文件夹
scene_dir = '/media/airs/BIN/cwah_ex/week2/gpt-4o-mini_20250221_143816/'
for scene in os.listdir(scene_dir):
    scene_path = os.path.join(scene_dir, scene)
    
    # 确保是目录并且名称以 "scene" 开头
    if os.path.isdir(scene_path) and scene.startswith('scene'):
        json_file_path = os.path.join(scene_path, 'refine_type_statistical_results.json')
        
        # 读取 JSON 文件
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                
                # 更新总次数
                for key, value in data.items():
                    if key in total_counts:
                        total_counts[key] += value
                    else:
                        total_counts[key] = value

# 统计总调用次数
total_calls = sum(total_counts.values())

# 计算每个步骤的频率
frequency = {key: value / total_calls for key, value in total_counts.items()}

# 创建 DataFrame 进行美观展示
df = pd.DataFrame({
    '调用次数': total_counts,
    '调用频率': frequency
}).reset_index().rename(columns={'index': '步骤'})

# 输出结果
print(df)
print("总调用次数:", total_calls)