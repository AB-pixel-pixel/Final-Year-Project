import os
import pandas as pd
import json
from pprint import pprint
baseline_average_prompt_tokens = 106037.1
baseline_average_completion_tokens = 42350.8
baseline_average_total_tokens = baseline_average_prompt_tokens + baseline_average_completion_tokens

# python visualize_result/cwah_token_framework.py
def calculate_averages(base_directory):
    total_time = 0
    scene_count = 0

    # Iterate through each scene folder
    for scene in os.listdir(base_directory):
        scene_path = os.path.join(base_directory, scene)
        if os.path.isdir(scene_path):
            # Path to the token_usage_log.csv file
            token_usage_file = os.path.join(scene_path, 'used_time.txt')
            print("token_usage_file: ", token_usage_file , end='')

            if os.path.isfile(token_usage_file):
                with open(token_usage_file, 'r') as file:
                    try:
                        used_time = float(file.read().strip())
                        print("used_time:",used_time)
                        total_time += used_time
                        scene_count += 1
                    except ValueError:
                        print(f"无法转换文件 {token_usage_file} 的内容为浮点数。")


    # Calculate averages
    average_time = total_time / scene_count if scene_count > 0 else 0
    print(f"Average time: {average_time}")


# Specify the base directory containing the scene folders
base_directory = '/media/airs/BIN/cwah_ex/week2/gpt-4o_20250311_103531'
calculate_averages(base_directory)