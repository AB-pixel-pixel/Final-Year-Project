import os
import json

def calculate_average_results(base_dir):
    total_finish = 0
    total_files = 0

    # Walk through all subdirectories in the base directory
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file == 'result_episode.json':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        total_finish += data.get('finish', 0)
                        total_files += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON from {file_path}")

    if total_files > 0:
        average_finish = total_finish / total_files
        print(f"Average finish: {average_finish}")
    else:
        print("No result_episode.json files found.")

if __name__ == "__main__":
    base_directory = '/media/airs/BIN/tdw_ex/running/try/LMs-gpt4o-mini_v2.01'
    calculate_average_results(base_directory)