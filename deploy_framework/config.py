import os
import time


current_environment = os.getenv('experiment_name', 'tdw')  # 默认值为 'tdw'
print("current_environment: ", current_environment)

if current_environment == 'tdw':
    DATA_ROOT_FOLDER = "/media/airs/BIN/tdw_ex/week2" # TODO change when nessary
else:
    DATA_ROOT_FOLDER = "/media/airs/BIN/cwah_ex/week2" # TODO change when nessary

# 用于读取top_down_image以构成视频的
k_ex_result_folder = "/media/airs/BIN/tdw_ex/results/try/LMs-inference_v3" # TOT"

tot_search_size = 1 # if size == 1, then don't use tot search
k_temperature = 0 # 0.7
k_top_p = 1

# 如果需要展示出每一步的思考过程则设置为True
display_inference_process = True
# 向大语言模型最多尝试请求次数，超过则不发送请求了，避免服务器过载，但是我这边还在不断发送请求,only settting for gpt model request
max_request_times = 3 # TODO
api_3_5_key = "sk-FQXU39vAmhDEkafk7f843657B13c4471Bc40Ee547592Ee5d"
api_key4_0_key = "sk-ZOpDZIXcs0LvE5VtE5D33cE14061425392487d59DeD7Ff71"


large_model_server_ip_port = "http://192.168.31.81:8800/api/generate" # "http://127.0.0.1:8030/api/generate" #  # 
model_name = 'gpt-4o-mini'  # "llama3.1" # "gpt-4o" # "gpt-4" 




timestamp = time.strftime("%Y%m%d_%H%M%S")
VERSION = f"{model_name}_{timestamp}"

k_current_experiment_data_folder = f"{DATA_ROOT_FOLDER}/{VERSION}"



# concat_image_saved_folder = f'{k_current_experiment_data_folder}/images/scene_{scene_id}'  # 替换为你的输出目录
# data_saved_path = os.path.join(k_current_experiment_data_folder, "data", f"scene{scene_id}_{VERSION}.json" )




# 运行过程中存放的数据
receive_send_msg_saved_path = f'input_output_log_{VERSION}.txt'# f'{k_current_experiment_data_folder}/input_output_log_{VERSION}.txt'

k_env = None

DEBUG = True
DEBUG_BLACKBOARD = False
PROGRESS_DEBUG = True
SAVE = True

# dynamic param during runtime
# 最大的尝试规划次数
MAX_PLANNING_TIMES = 2
explore_exploitation_config = {
    "threshold": 3,
    "window_size" : 4,
}
k_exemption_valid_time = 8


