import os
import json

# 定义路径
base_paths = ['/media/airs/BIN/tdw_ex/results/try/LMs-test2','/media/airs/BIN/tdw_ex/results_test/try/LMs-test2']
success_rates = []
for base_path in base_paths:
    # 遍历下级文件夹
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            result_file_path = os.path.join(folder_path, 'result_episode.json')
            print("result_file_path: ",result_file_path)
            # 检查文件是否存在
            if os.path.isfile(result_file_path):
                with open(result_file_path, 'r') as file:
                    data = json.load(file)
                    finish = data.get('finish', 0)
                    total = data.get('total', 1)  # 防止除以零
                    success_rate = finish / total
                    # 将场景编号转换为整数并存储

                    success_rates.append((int(folder), success_rate))

# 按场景编号排序
success_rates.sort(key=lambda x: x[0])

# 打印结果
for scene, rate in success_rates:
    print(f"{scene}  success_rate : {rate}")

# 计算平均成功率
if success_rates:
    average_success_rate = sum(rate for _, rate in success_rates) / len(success_rates)
else:
    average_success_rate = 0

# 打印成功率列表和平均成功率
print("成功率列表:", [rate for _, rate in success_rates])
print("平均成功率:", average_success_rate)