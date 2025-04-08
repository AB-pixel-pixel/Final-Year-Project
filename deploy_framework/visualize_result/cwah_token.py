import os
import pandas as pd

# 设置文件夹路径
folder_path = "/media/airs/BIN/baseline/result_from_REVECA_AAAI-25"
experiment_name = 'statistical_time_consumption'
folder_path = "/media/airs/BIN/baseline/result_from_REVECA_AAAI-25/statistical_time_consumption/LLMs_comm_test1_gpt-4o-mini"# os.path.join(folder_path, experiment_name)
output_file = os.path.join(folder_path, 'summary_statistics.csv')

# 初始化一个空的 DataFrame
summary_data = []

# 遍历文件夹中的所有CSV文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # 构造文件路径
        file_path = os.path.join(folder_path, filename)
        print("reading csv file:", file_path)
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 计算统计数据
        scene_name = filename.split('_')[-1].replace('.csv', '')
        input_tokens_sum = df['input_tokens'].sum()
        output_tokens_sum = df['output_tokens'].sum()
        total_tokens_sum = df['total_tokens'].sum()
        cost_sum = df['cost'].sum()
        
        # 添加到统计数据列表
        summary_data.append([scene_name, input_tokens_sum, output_tokens_sum, total_tokens_sum, cost_sum])

# 创建DataFrame
summary_df = pd.DataFrame(summary_data, columns=['scene_name', 'input_tokens', 'output_tokens', 'total_tokens', 'cost'])

# 计算平均值
averages = summary_df.mean(numeric_only=True)
averages['scene_name'] = 'Average'

# 将平均值添加到数据框
summary_df = summary_df.append(averages, ignore_index=True)

# 将结果写入CSV文件
summary_df.to_csv(output_file, index=False)

print(f"统计数据已输出到 {output_file}")