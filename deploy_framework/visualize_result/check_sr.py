import sys
import json
import os
from prettytable import PrettyTable
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import k_ex_result_folder
# 定义文件夹路径

csv_file_path = os.path.join(k_ex_result_folder, "results_summary.csv")  # CSV 文件保存路径

# 创建表格
table = PrettyTable(["场景编号", "成功搬运物体数量", "需要搬运物体数量", "成功率"])
success_rates = []
# 遍历子文件夹
for i in range(0,24):
    folder_path = f"{k_ex_result_folder}"
    subfolder_name = str(i)
    file_path = os.path.join(folder_path, subfolder_name, "result_episode.json")

    try:
        # 打开文件并读取JSON数据
        with open(file_path, 'r') as f:
            data = json.load(f)

        # 获取数据
        finish = data["finish"]
        total = data["total"]
        success_rate = finish / total if total != 0 else 0  # 防止除以零错误
        success_rates.append(success_rate)
        # 添加到表格
        table.add_row([subfolder_name, finish, total, f"{success_rate:.2%}"])

    except FileNotFoundError:
        table.add_row([subfolder_name, "N/A", "N/A", "N/A"])
        # print(f"实验 {subfolder_name} 的 result_episode.json 文件未找到.{file_path}")
    except json.JSONDecodeError:
        table.add_row([subfolder_name, "N/A", "N/A", "N/A"])
        print(f"实验 {subfolder_name} 的 result_episode.json 文件格式错误.")
    except KeyError as e:
        table.add_row([subfolder_name, "N/A", "N/A", "N/A"])
        print(f"实验 {subfolder_name} 的 result_episode.json 文件缺少键值: {e}")
if len(success_rates):
    table.add_row(["平均成功率", "-", "-", f"{sum(success_rates)/len(success_rates):.2%}"])
    print(table)
else:
    print("")

# 打印表格


# 保存到 CSV 文件
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["场景编号", "成功搬运物体数量", "需要搬运物体数量", "成功率"])
    
    for row in table.rows:
        writer.writerow(row)

    # 写入平均成功率
    writer.writerow(["平均成功率", "-", "-", f"{sum(success_rates)/len(success_rates):.2%}"])

