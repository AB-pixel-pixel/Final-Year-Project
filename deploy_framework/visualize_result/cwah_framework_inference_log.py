import os
import pandas as pd
import json
from pprint import pprint
baseline_average_prompt_tokens = 106037.1
baseline_average_completion_tokens = 42350.8
baseline_average_total_tokens = baseline_average_prompt_tokens + baseline_average_completion_tokens

# python visualize_result/cwah_token_framework.py
def calculate_averages(base_directory):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_inference_time = 0
    scene_count = 0

    # Iterate through each scene folder
    for scene in os.listdir(base_directory):
        scene_path = os.path.join(base_directory, scene)
        print("scene_path: ",scene_path)
        if os.path.isdir(scene_path):
            # Path to the token_usage_log.csv file
            token_usage_file = os.path.join(scene_path, 'token_usage_log.csv')

            if os.path.isfile(token_usage_file):
                # Read the CSV file
                df = pd.read_csv(token_usage_file)

                # Sum tokens
                total_prompt_tokens += df['Prompt Tokens'].sum()
                total_completion_tokens += df['Completion Tokens'].sum()
                total_inference_time += df['Inference Time'].sum()
                scene_count += 1
                print("scene count")

    # Calculate averages
    average_prompt_tokens = total_prompt_tokens / scene_count if scene_count > 0 else 0
    average_completion_tokens = total_completion_tokens / scene_count if scene_count > 0 else 0
    average_inference_time = total_inference_time / scene_count if scene_count > 0 else 0

    # Prepare results
    results = {
        'Average Prompt Tokens': average_prompt_tokens,
        'Average Prompt Tokens percentage' : average_prompt_tokens / baseline_average_prompt_tokens ,
        'Average Completion Tokens': average_completion_tokens,
        'Average Completion Tokens percentage' : average_completion_tokens / baseline_average_completion_tokens ,
        'Average Total Tokens': average_completion_tokens + average_prompt_tokens,
        'Average Total Tokens percentage' : (average_completion_tokens + average_prompt_tokens) / baseline_average_total_tokens,
        'Average Inference Time' : average_inference_time,
        'Scene Count': scene_count
    }
    pprint(results)
    # Save results to a JSON file
    # output_file = os.path.join(base_directory, 'token_averages.json')
    # with open(output_file, 'w') as json_file:
    #     json.dump(results, json_file, indent=4)

    # print(f"Averages saved to {output_file}")

# Specify the base directory containing the scene folders
base_directory =  '/media/airs/BIN/cwah_ex/week2/qwen2.5-7B_20250319_124904'
calculate_averages(base_directory)