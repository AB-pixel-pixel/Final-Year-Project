import os
import pickle

# 文件夹路径
folder_path = "/media/airs/BIN/baseline/time_counter_from_REVECA_gpt4o"

time_sum = 0
file_count = 0
# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_count += 1
        file_path = os.path.join(folder_path, filename)
        # 读取 .pik 文件
        with open(file_path, 'rb') as file:
            # 改為讀取.txt文件
            data = file.read()
            time_sum += float(data)
print("平均用時:", time_sum / file_count)