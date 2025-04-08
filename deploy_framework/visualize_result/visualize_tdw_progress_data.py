import os
import numpy as np
import sys
import copy
from datetime import datetime

import json

import os
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import k_current_experiment_data_folder
from LOG_SYSTEM import LOG_SYSTEM

def get_lastest_action_info(action_info:dict):
    action_info_  = copy.deepcopy(action_info)
    action_info_["length"] = 1
    action_info_["log"] = [action_info_["log"][-1]]
    return action_info_


def convert_dict(temp :dict):
    return {int(k):v for k,v in temp.items()}


def normalize_frame_info(frame_info : list = []):
    # self.each_frame_information.append({"num_frames": num_frames, "actions_description": actions_description, "current_task": current_task,"image_path": path})
    global_data = {info["num_frames"]: [info["image_path"]] for info in frame_info}
    return global_data



def process_text(pipeline_planning_data : list , frame_info: list = [], data_saved_path: str = None):
    # self.planning_log[step_num][agent_id] = {'obs':obs,'log':[(plan,counter,planning_stage,reason)], 'length': 1}
    frame_info_ = normalize_frame_info(frame_info)
    visualize_data = convert_dict(frame_info_)
    # pipeline_planning_data = convert_dict(pipeline_planning_data)

    
    agents_state = [{},{}]
    agents_latest_state = [{},{}]
    # frame_info_ key is str number not int number 
    for step_num in frame_info_.keys():
        step_num = int(step_num)
        planning_data = pipeline_planning_data.get(step_num)
        if planning_data:
            planning_data = convert_dict(planning_data)
            for agent_id in [0,1]:

                if agent_id in planning_data:
                    agent_info = planning_data[agent_id]
                    agents_state[agent_id] = agent_info
                    agents_latest_state[agent_id] = get_lastest_action_info(agent_info)
                    visualize_data[step_num].append(copy.deepcopy(agent_info))
                else:
                    agent_info = agents_latest_state[agent_id]
                    visualize_data[step_num].append(copy.deepcopy(agent_info))
        else:
            visualize_data[step_num].extend(copy.deepcopy(agents_latest_state))
        visualize_data[step_num].append(frame_info[step_num].get("current_task"))
    

    with open(data_saved_path, 'w',encoding='utf-8') as json_file:
        json.dump(visualize_data, json_file, indent=4)
    print(f"Data saved to {data_saved_path}")
    return data_saved_path


def visualize_tdw_process_data_for_web(
                                pipeline_planning_data : list = [],each_frame_information_path: str =  "",data_saved_path : str = ""):
    
    os.makedirs(k_current_experiment_data_folder, exist_ok=True)
    with open(each_frame_information_path, 'r') as json_file:
        # self.each_frame_information.append({"num_frames": num_frames, "actions_description": actions_description, "current_task": current_task,"image_path": path})
        frame_info = json.load(json_file)

    return process_text(pipeline_planning_data = pipeline_planning_data,frame_info = frame_info, data_saved_path = data_saved_path)



if __name__ == '__main__':

    with open(r"/media/airs/BIN/cwah_ex/cwah_planning_data/scene26_v5_20250109_143736.log", 'r',encoding='utf-8') as file:
        pipeline_planning_data = json.load(file)

    with open(r"/media/airs/BIN/cwah_ex/frame_info_from_cwah/scene_0_version_v4_None_frame_path.json", 'r',encoding='utf-8') as file:
        image_folder_info = json.load(file)
    # visualize_tdw_process_data_for_web(
    #     image_folder_info=image_folder_info,
    #     scene_id=26,
    #     pipeline_planning_data = pipeline_planning_data
    # )