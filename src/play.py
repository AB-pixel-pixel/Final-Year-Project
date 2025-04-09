import json
import time
# 檔案路徑
file_path = "/media/airs/BIN/tdw_ex/training_data/try/LMs-test_rl_v1/23/each_frame_information.json"

# 讀取 JSON 檔案
with open(file_path, 'r') as file:
    log_data = json.load(file)

# 輸出讀取的資料
# print(log_data)
# 自定義幀率
fps = 24  # 每秒播放的幀數
interval = 1 / fps  # 每幀的間隔時間

# 合併內容相同的幀
unique_frames = []
previous_actions = ""
previous_task = ""
for entry in log_data:
    if entry["actions_description"] != previous_actions or entry["current_task"] != previous_task:
        unique_frames.append(entry)
        previous_actions = entry["actions_description"]
        previous_task = entry['current_task']
# 播放日誌
for frame in unique_frames:
    print(f": {frame['num_frames']},\nActions: \t {frame['actions_description']}\n Task:\t {frame['current_task']}\n") # Image: {frame['image_path']}
    time.sleep(interval)  # 根據 FPS 暫停