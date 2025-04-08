import pickle
import os
# 定义文件路径
file_path = '/media/airs/BIN/baseline/result_from_REVECA_AAAI-25/statistical_time_consumption/LLMs_comm_gpt-4o-mini/results.pik'

file_path = '/media/airs/BIN/cwah/cwah/test_results/framework_test39_gpt4o_mini'
file_path = os.path.join(file_path, 'results.pik')


# 读取并打印结果
with open(file_path, 'rb') as file:
    results = pickle.load(file)
    print(results)

    # 计算 'L' 值的平均数
    total_L = 0
    count = 0
    
    for key, value in results.items():
        total_L += value['L'][0]  # 假设 'L' 是一个列表，取第一个元素
        count += 1
        is_finished = value["S"][0]
        if is_finished == True or is_finished == 1:
            print(f"场景{key} 任务成功: 消耗步长{value['L'][0]}")
        else:
            print(f"场景{key} 任务失败: 消耗步长{value['L'][0]}")

    average_L = total_L / count if count > 0 else 0
    print("平均步长为:", average_L)