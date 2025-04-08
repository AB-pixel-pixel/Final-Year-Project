from argparse import OPTIONAL
import os
import time
# 写入环境变量
os.environ['experiment_name'] = 'cwah'



import sys
import json
from typing import Optional

from fastapi import FastAPI,HTTPException
from framework_structure import LOG_SYSTEM
from config import receive_send_msg_saved_path , k_env
from pipeline import Pipeline,log_system
from communication_protocol import BATCH_PROMPT_Constituent_Elements,SceneName,cwah_image_saved_path, cwah_steps
from visualize_result.visualize_cwah_progress_data import visualize_cwah_process_data_for_web, visualize_symbolic_cwah_process_data_for_web
from config import k_current_experiment_data_folder

import pyfiglet
print(pyfiglet.figlet_format("CWAH"))

from cwah_time_token import calculate_averages




# 开启服务
# uvicorn framework_server_cwah:app --reload
# uvicorn framework_server_cwah:app --port 8000
# uvicorn framework_server_cwah:app --port 8001
# uvicorn framework_server_cwah:app --port 8002
# uvicorn framework_server_cwah:app --port 8003
# uvicorn framework_server_cwah:app --port 8004

# 重启服务
# curl -X POST http://127.0.0.1:8000/api/restart 
# curl -X POST -v http://127.0.0.1:8000/api/restart

# 关闭服务
# curl -X POST -v http://127.0.0.1:8000/api/close
# curl -X POST -H "Content-Type: application/json" -d '{"steps": 5}' http://127.0.0.1:8001/api/close_symbolic


pipeline : Optional[Pipeline] = None
scene_id = None

app = FastAPI()

start_time = 0.0
def input_log(batch_prompt: BATCH_PROMPT_Constituent_Elements):
    try:
        with open(receive_send_msg_saved_path, 'a', encoding='utf-8') as f:
            f.write("--------------prompt--------------\n")
            f.write(str(batch_prompt.model_dump()))  # 将整个 prompt 对象写入
            f.write('\n')
        print(f"input 成功追加写入到: {receive_send_msg_saved_path}")
    except Exception as e:
        print(f"追加写入失败: {e}")

def output_text_log(batch_action):
    try:
        with open(receive_send_msg_saved_path, 'a', encoding='utf-8') as f:
            f.write("--------------ans--------------\n")
            f.write(str(batch_action))
            f.write('\n')
    except Exception as e:
        log_system.PRINT(f"追加写入失败: {e}")



@app.post("/api/think")
def think(batch_prompt: BATCH_PROMPT_Constituent_Elements):
    # Function body forrbide LOG()

    log_system.PRINT("thinking ")
    input_log(batch_prompt)
    for elements in batch_prompt.batch_prompt_constituent_elements:
        log_system.LOG(f"输入信息: {elements.model_dump()}")  # 打印完整的 prompt 信息
    # try:
    batch_action = pipeline.receive(batch_prompt) # batch_action : List[str]
    # except Exception as e:
    #     log_system.PRINT(e)
    #     pipeline.close()  # 关闭现有的 pipeline
        # sys.exit(f"scene {LOG_SYSTEM.SCENE_NAME} 实验结束，异常退出")


    log_system.LOG(f"输出信息: {batch_action}")

    log_system.LOG(f"\n\n########################################\n\n")
    print(f"输出信息: {batch_action}")
    output_text_log(batch_action)

    return batch_action


@app.post("/api/start")
def start(data : SceneName):
    log_system.update(data.scene_name)
    global start_time
    start_time = time.time()
    global pipeline
    pipeline = Pipeline()  # 重新创建一个新的 pipeline
    global scene_id
    scene_id = data.scene_name
    output_text_log(f"-------------- scene {data.scene_name} --------------\n")

    return {"message": f"Start! Scene name set to {data.scene_name}"}



@app.post("/api/close_symbolic")
def close_symbolic_pipeline(data: cwah_steps):
    # global pipeline
    log_system.PL("data.steps\n",data.steps)# KEY STEPS_NUM, VALUE IS [IMAGE0_PATH, IMAGE1_PATH]
    used_time = time.time()- start_time
    log_system.log_used_time(used_time)
    planning_data = pipeline.close()  # 关闭现有的 pipeline
    data_saved_path = visualize_symbolic_cwah_process_data_for_web(pipeline_planning_data = planning_data, total_step = data.steps, saving_path = log_system.display_log_file)


    log_system.PL("data_saved_path\n",data_saved_path)
    log_system.PRINT("Pipeline close successfully.")

    print("k_current_experiment_data_folder: ", k_current_experiment_data_folder)
    calculate_averages(k_current_experiment_data_folder)
    return {"message": "Pipeline restarted successfully."}


# @app.post("/api/close")
# def close_pipeline(data: cwah_image_saved_path):
#     # global pipeline
#     log_system.PL("data.path\n",data.path)# KEY STEPS_NUM, VALUE IS [IMAGE0_PATH, IMAGE1_PATH]
#     planning_data = pipeline.close()  # 关闭现有的 pipeline

#     with open(data.path, 'r') as file:
#         image_folder_info_data = json.load(file)
#     data_saved_path = visualize_cwah_process_data_for_web(image_folder_info = image_folder_info_data,scene_id = scene_id,pipeline_planning_data = planning_data)

#     log_system.PL("data_saved_path\n",data_saved_path)
#     log_system.PRINT("Pipeline close successfully.")