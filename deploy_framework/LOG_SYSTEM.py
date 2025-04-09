
import asyncio
import inspect
import os
import time
import csv
from config import VERSION , k_current_experiment_data_folder
import json 
import textwrap
import pandas as pd
from typing import Dict



# ---------------------------- 日志系统 ------------------------

print("local local")

class LOG_SYSTEM:
    LOG_FILE = None 
    CSV_FILE = None
    SCENE_NAME = "-1"
    EX_NAME = ""
    VERSION = VERSION
    CURRENT_DIALOG_COUNT = 0
    CURRENT_TOTAL_TOKENS = 0
    TOTAL_COST_MONEY = 0
    ACCUMULATE_TIME = 0
    SOURCE = None
    log_file_lock = asyncio.Lock()
    token_log_file_lock = asyncio.Lock()
    LINE_WIDTH = 130
    STAGE_TIME_CONSUMPTION : Dict = dict()

    STAGE_TIME_CONSUMPTION_SAVING_FILE : str = ""

    
    def __init__(self):

        self.planning_log = dict() # saving the planning process
        self.log_dir : str = ""
        self.refine_type_counter : dict = {}
        self.display_log_file : str = "" # using in cwah env, not tdw
    def update(self,scene_name):

        LOG_SYSTEM.SCENE_NAME = scene_name
        LOG_SYSTEM.EX_NAME = f"scene{LOG_SYSTEM.SCENE_NAME}"
        ex_name = LOG_SYSTEM.EX_NAME
        
        self.log_dir = os.path.join(k_current_experiment_data_folder, ex_name)
        os.makedirs(self.log_dir , exist_ok=True)
        
        LOG_SYSTEM.LOG_FILE = os.path.join(self.log_dir, "framework_log.txt")
        LOG_SYSTEM.CSV_FILE = os.path.join(self.log_dir, "token_usage_log.csv")
        LOG_SYSTEM.DISPLAY_RAW_LOG_FILE = os.path.join(self.log_dir, "display_raw_data.txt")
        LOG_SYSTEM.PLAN_LOG_FILE = os.path.join(self.log_dir, "plan_data.txt")
        LOG_SYSTEM.REFINE_TYPE_STATISTICAL_RESULTS_FILE =  os.path.join(self.log_dir, "refine_type_statistical_results.json")
        LOG_SYSTEM.STAGE_TIME_CONSUMPTION = dict()
        LOG_SYSTEM.STAGE_TIME_CONSUMPTION_SAVING_FILE = os.path.join(self.log_dir, "stage_time_consumption.csv")
        self.display_log_file = os.path.join(self.log_dir, "display_log.json")

       
        LOG_SYSTEM.CURRENT_DIALOG_COUNT = 0
        LOG_SYSTEM.CURRENT_TOTAL_TOKENS = 0
        LOG_SYSTEM.TOTAL_COST_MONEY = 0
        LOG_SYSTEM.ACCUMULATE_TIME = 0
        self.planning_log = dict()
        self.refine_type_counter = dict()

        LOG_SYSTEM.USED_TIME_LOG_PATH = os.path.join(self.log_dir, "used_time.txt")
    def log_used_time(self,used_time: float):
        with open(LOG_SYSTEM.USED_TIME_LOG_PATH, "a", encoding="utf-8") as f:  # 使用追加模式打开文件
            f.write(str(used_time))
            f.write("\n")

    def refine_call_counts(self, refine_type : str):
        """ Used to count the number of calls to different refine type functions and their respective frequencies. """
        self.refine_type_counter[refine_type] = self.refine_type_counter.get(refine_type, 0) + 1
    
    # logging function 
    def add_log(self, public_pipeline_state: "AGENT_STATE" , agent, supplementary_explanation: str = ""):
        """ Saving the planning data to display the overall planning process, """
        plan = None

        if agent.action_tree is None or agent.action_tree.head_node is None or agent.action_tree.head_node.children == []:
            plan = "None"
        else:
            plan = agent.action_tree.get_overall_plan_in_text()
        self._add_log(step_num=public_pipeline_state.get("step_num"),
                    obs=agent.state.get('observation'),
                    plan=plan,
                    counter=0, 
                    agent_id=agent.agent_id,
                    planning_stage="",
                    current_task=public_pipeline_state.get("current_task"),
                    supplementary_explanation = supplementary_explanation)


    def _add_log(self, step_num:int = 0, obs:str = "", 
                plan: str = "", counter:int = 0, 
                planning_stage: str = "" , agent_id:int = 0, supplementary_explanation:str = "", current_task: str = ""):
        step_num = int(step_num)
        if not step_num in self.planning_log:
            self.planning_log[step_num] = dict()

        if not agent_id in self.planning_log[step_num]:
            self.planning_log[step_num][agent_id] = {'obs':obs,'log':[(plan,counter,planning_stage,supplementary_explanation,current_task)], 'length': 1}
        else:
            self.planning_log[step_num][agent_id]['log'].append((plan,counter,planning_stage,supplementary_explanation,current_task))
            self.planning_log[step_num][agent_id]['length'] += 1
            
    def save_log(self):
        if LOG_SYSTEM.DISPLAY_RAW_LOG_FILE:
            with open(LOG_SYSTEM.DISPLAY_RAW_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.planning_log, f, ensure_ascii=False, indent=4)
                print(f"saving planning log to {LOG_SYSTEM.DISPLAY_RAW_LOG_FILE}")
        else:
            print("日志文件未定义")
        return self.planning_log

    def PRINT(self,*args, **kwargs):
        """
        打印内容并包含代码行号，类似内置的print函数。
        """
        caller_frame = inspect.currentframe().f_back
        line_number = caller_frame.f_lineno
        filename = caller_frame.f_code.co_filename
        print(f"{filename}:{line_number} - ",*args, **kwargs,end="\n") # 再打印用户提供的参数

    def LOG(self,*args):
        """
        将消息记录到日志文件。

        Args:
            *args: 要记录的消息，可以是多个参数。
        """
        if LOG_SYSTEM.LOG_FILE:
            with open(LOG_SYSTEM.LOG_FILE, "a", encoding="utf-8") as f:  # 使用追加模式打开文件
                for arg in args:  # 遍历所有参数并写入文件
                    f.write(str(arg))
                f.write("\n")
        # else:
        #     print("日志文件未定义")

    def PL(self,*args, **kwargs):
        """
        打印内容并记录到日志文件。

        Args:
            *args: 要打印和记录的消息，可以是多个参数。
        """
        caller_frame = inspect.currentframe().f_back
        line_number = caller_frame.f_lineno
        filename = caller_frame.f_code.co_filename
        print(f"{filename}:{line_number} - ", end="")  # 先打印文件名和行号
        print(*args, **kwargs) # 再打印用户提供的参数
        # print # 打印一个空行，使输出更美观
        
        text = "".join([str(arg) for arg in args])
        if LOG_SYSTEM.LOG_FILE:
            with open(LOG_SYSTEM.LOG_FILE, "a", encoding="utf-8") as f:  # 使用追加模式打开文件
                f.write(text)

                f.write("\n")


    def log_planning_data(self,text : str):
        if LOG_SYSTEM.PLAN_LOG_FILE:
            with open(LOG_SYSTEM.PLAN_LOG_FILE, "a", encoding="utf-8") as f:  # 使用追加模式打开文件
                f.write(text)
                f.write("\n")

    # --------------------------- 记录token用量 ---------------------------

    async def log_token_usage(self,usage, dialog_type , content = "",elapsed_time_rounded : float = 0):
        """
        向 CSV 文件中追加 token 使用记录。
        """
        async with self.log_file_lock:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            increment_total_tokens = usage.total_tokens
            inference_time = elapsed_time_rounded
            LOG_SYSTEM.TOTAL_COST_MONEY += usage.total_cost
            LOG_SYSTEM.ACCUMULATE_TIME += inference_time

            if not os.path.exists(LOG_SYSTEM.CSV_FILE):
                with open(LOG_SYSTEM.CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(["Timestamp", "Total Dialogs", "Token Source", "Content", "Prompt Tokens", "Completion Tokens","Increment", "Total Tokens"," Cost" , "Inference Time","Accumulate Time"])
                # 获取当前时间戳，并格式化
                

            LOG_SYSTEM.CURRENT_TOTAL_TOKENS += increment_total_tokens
            LOG_SYSTEM.CURRENT_DIALOG_COUNT += 1
            with open(LOG_SYSTEM.CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, 
                                LOG_SYSTEM.CURRENT_DIALOG_COUNT, 
                                dialog_type, 
                                content, 
                                prompt_tokens, 
                                completion_tokens,
                                increment_total_tokens, 
                                LOG_SYSTEM.CURRENT_TOTAL_TOKENS,
                                LOG_SYSTEM.TOTAL_COST_MONEY,
                                inference_time,
                                LOG_SYSTEM.ACCUMULATE_TIME
                                ])

            if LOG_SYSTEM.STAGE_TIME_CONSUMPTION.get(dialog_type) is None:
                LOG_SYSTEM.STAGE_TIME_CONSUMPTION[dialog_type] = [{"inference_time":inference_time,"counter": 1}]
            else:
                temp_count = LOG_SYSTEM.STAGE_TIME_CONSUMPTION[dialog_type][-1]["counter"]
                LOG_SYSTEM.STAGE_TIME_CONSUMPTION[dialog_type].append({"inference_time":inference_time,"counter": temp_count+1})


    def display_token_usage(self):
        """
        展示当前 token 使用情况。

        Args:
            csv_file (str): CSV 文件的路径。默认为 "token_usage_log.csv"。
        """
        if not os.path.exists(LOG_SYSTEM.CSV_FILE):
            self.PRINT("CSV file does not exist.")
            return 
        df = pd.read_csv(LOG_SYSTEM.CSV_FILE)
        self.PRINT(df)

    def close(self):
        """
        关闭日志系统。保存一整组实验每个阶段的耗时
        """
        self.PRINT("Closing log system.")
        with open(LOG_SYSTEM.STAGE_TIME_CONSUMPTION_SAVING_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Stage", "Function Call Times","Accumulated Inference Time","Average Time per Call"])
            for dialog_type, values in LOG_SYSTEM.STAGE_TIME_CONSUMPTION.items():
                total_inference_time = sum(item["inference_time"] for item in values)
                total_counter = sum(item["counter"] for item in values)
                writer.writerow([str(dialog_type),str(total_counter),str(total_inference_time),round(total_inference_time/total_counter,4)])

        self.PL("Token 消耗用量日志路径: ", LOG_SYSTEM.CSV_FILE)
        self.PL("Framework 运行日志路径: ", LOG_SYSTEM.LOG_FILE)
        self.PL("展示数据路径: ", LOG_SYSTEM.DISPLAY_RAW_LOG_FILE)
        self.PL("规划日志路径: ", LOG_SYSTEM.PLAN_LOG_FILE)
        self.PL("用时日志已经保存到文件: " , LOG_SYSTEM.STAGE_TIME_CONSUMPTION_SAVING_FILE)
        self.PL("Refine类型统计结果: ",LOG_SYSTEM.REFINE_TYPE_STATISTICAL_RESULTS_FILE)
        # 保存REFINE 类型的函数


        with open(LOG_SYSTEM.REFINE_TYPE_STATISTICAL_RESULTS_FILE, mode='w', encoding='utf-8') as f:
            json.dump(self.refine_type_counter, f, ensure_ascii=False, indent=4)
# ---------------------------- 日志系统 END ------------------------
log_system = LOG_SYSTEM()