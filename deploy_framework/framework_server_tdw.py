
import os
import sys
import signal
from typing import Optional, Literal, Annotated
import pyfiglet
import time
# 写入环境变量
os.environ['experiment_name'] = 'tdw'
print(pyfiglet.figlet_format("TDW SERVER"))



from fastapi import FastAPI, APIRouter

from framework_structure import LOG_SYSTEM,frame_saved_path
from communication_protocol import SceneName
from pipeline import Pipeline,log_system,BATCH_PROMPT_Constituent_Elements
from visualize_result.visualize_tdw_progress_data import visualize_tdw_process_data_for_web
from cwah_time_token import calculate_averages

# 开启服务
# uvicorn framework_server_tdw:app --reload
# uvicorn framework_server_tdw:app --port 8000
# uvicorn framework_server_tdw:app --port 8001
# 重启服务
# curl -X POST http://127.0.0.1:8000/api/restart 
# curl -X POST -v http://127.0.0.1:8000/api/restart

# 关闭服务
# curl -X POST -v http://127.0.0.1:8000/api/close
# curl -X POST -v http://127.0.0.1:8001/api/close

is_closed : bool = True
pipeline : Pipeline = Pipeline() 
app = FastAPI()
router = APIRouter()


# 处理 Ctrl + C
def signal_handler(sig, frame):
    if not is_closed:
        log_system.PRINT("Received shutdown signal. Saving pipeline data. ")
        if pipeline:
            pipeline.close()  # 关闭现有的 pipeline
    else:
        log_system.PRINT("Received shutdown signal. Pipeline already shutdown.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


@router.post("/api/think")
def think(batch_prompt: BATCH_PROMPT_Constituent_Elements):
    # Function body forrbide LOG()
    global is_closed
    is_closed = False

    path = 'prompt_from_tdw_mat_scene_week2.txt'
    log_system.PRINT("thinking ")
    with open(path, 'a', encoding='utf-8') as f:
        f.write("--------------prompt--------------\n")
        f.write(str(batch_prompt.model_dump()))  # 将整个 prompt 对象写入
        f.write('\n')
    log_system.PRINT(f"成功追加写入到: {path}")

    for elements in batch_prompt.batch_prompt_constituent_elements: 
        log_system.LOG(f"输入信息: {elements.model_dump()}")  # 打印完整的 prompt 信息

    #try:
    batch_action = pipeline.receive(batch_prompt) # batch_action : List[str]
    # except Exception as e:
    #     log_system.PRINT(e)
    #     log_system.PRINT(f"scene {LOG_SYSTEM.SCENE_NAME} 实验出错，请检查日志")
    #     pipeline.close()  # 关闭现有的 pipeline
    #     return None
    #     # sys.exit(f"scene {LOG_SYSTEM.SCENE_NAME} 实验出错，请检查日志")
        

    log_system.PRINT(f"输出信息: {batch_action}")
    log_system.LOG(f"输出信息: {batch_action}")

    with open(path, 'a', encoding='utf-8') as f:
        f.write("--------------ans--------------\n")
        f.write(str(batch_action))
        f.write('\n')
   

    # log_system.PRINT("type(batch_action): ",type(batch_action))
        # 验证输入的数据是否符合 Answer 模型
    return batch_action


@router.post("/api/start")
def set_scene_name(data : SceneName):
    global start_time, is_closed
    is_closed = False
    start_time = time.time()
    log_system.update(data.scene_name)
    global pipeline

    pipeline = Pipeline()  # 重新创建一个新的 pipeline

    return {"message": f"Scene name set to {data.scene_name}"}


@router.post("/api/close")
def close_pipeline(data: frame_saved_path):
    global is_closed
    is_closed = True
    # global pipeline
    used_time = time.time()- start_time

    pipeline_planning_data = pipeline.close()  # 关闭现有的 pipeline
    result_path = visualize_tdw_process_data_for_web(
        pipeline_planning_data = pipeline_planning_data,
        each_frame_information_path = data.path,
        data_saved_path = log_system.display_log_file
        )
    
    print(f"visualize data save in to {result_path}")


    log_system.PRINT("Pipeline restarted successfully.")
    from config import k_current_experiment_data_folder
    print("k_current_experiment_data_folder: ", k_current_experiment_data_folder)
    calculate_averages(k_current_experiment_data_folder)

    return {"message": "Pipeline restarted successfully."}



app.include_router(router)


