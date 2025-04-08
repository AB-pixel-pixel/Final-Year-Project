import os
import json
from config import k_ex_result_folder



def calculate_completion_ratio(directory):
    results = {}
    total_ratio = 0
    count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'result_episode.json':
                file_path = os.path.join(root, file)
                if VERSION in file_path:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        finish = data.get('finish', 0)
                        total = data.get('total', 0)
                        if total > 0:
                            ratio = finish / total
                            total_ratio += ratio
                            count += 1
                        else:
                            ratio = 0
                        results[root] = ratio
    
    average_ratio = total_ratio / count if count > 0 else 0
    return results, average_ratio

if __name__ == "__main__":
    completion_ratios, average_completion_ratio = calculate_completion_ratio(k_ex_result_folder)
    
    for folder, ratio in completion_ratios.items():
        if VERSION in folder:
            print(f"文件夹: {folder}, 完成比例: {ratio:.2%}")
    
    print(f"平均完成比例: {average_completion_ratio:.2%}")