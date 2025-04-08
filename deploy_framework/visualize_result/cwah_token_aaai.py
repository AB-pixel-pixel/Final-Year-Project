import os
import pandas as pd

# 文件夹路径
folder_path = '/media/airs/BIN/baseline/result_from_REVECA_AAAI-25/statistical_time_consumption1' # '/media/airs/BIN/baseline/time_counter_from_REVECA'
# 用于存储所有浮点数的列表
values = []
cnt = 0
tokens_sum = 0
# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print("file : ",file_path)
        # 读取文件中的浮点数
        data = pd.read_csv(file_path)
        # print("data : ",data)
        cnt += 1
        tokens_sum += data['input_tokens'].sum()+ data['output_tokens'].sum()
        
# 计算平均值

average_value = tokens_sum / cnt
print(f'平均值: {average_value}')
