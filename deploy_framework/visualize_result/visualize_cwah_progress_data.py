import cv2
import os
import re
import numpy as np
import time
import textwrap
import sys
import copy
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import VERSION,k_current_experiment_data_folder

import json

import os
from PIL import Image

def get_lastest_action_info(action_info:dict):
    action_info_  = copy.deepcopy(action_info)
    action_info_["length"] = 1
    action_info_["log"] = [action_info_["log"][-1]]
    return action_info_

def concat_images(image_folder_info:dict, output_dir):
    # image_folder_info
    # KEY STEPS_NUM, VALUE IS [IMAGE0_PATH, IMAGE1_PATH]

    frame_info = {}
    # 处理图像
    for frame, image_paths in image_folder_info.items():
        # 检查左右图像是否存在
        left_image_path = os.path.join("/media/airs/BIN/cwah/cwah",image_paths[0])
        right_image_path = os.path.join("/media/airs/BIN/cwah/cwah",image_paths[1])
        if os.path.exists(left_image_path) and os.path.exists(right_image_path):
            # 打开图像
            left_image = Image.open(left_image_path)
            right_image = Image.open(right_image_path)

            # 创建白色条
            white_strip = Image.new('RGB', (20, max(left_image.height, right_image.height)), (255, 255, 255))

            # 拼接图像
            new_image = Image.new('RGB', (left_image.width + right_image.width + white_strip.width, left_image.height))
            new_image.paste(left_image, (0, 0))
            new_image.paste(white_strip, (left_image.width, 0))
            new_image.paste(right_image, (left_image.width + white_strip.width, 0))

            # 保存新图像
            new_image_path = os.path.join(output_dir, f"{frame}.png")
            new_image.save(new_image_path)

            frame_info[frame] = [os.path.abspath(new_image_path)]
        else:
            print("路径不存在")
    return frame_info # frame_info[frame] = [os.path.abspath(new_image_path)]


def convert_dict(temp :dict):
    return {int(k):v for k,v in temp.items()}

def process_text(pipeline_planning_data :dict = dict(), frame_info_:dict = {}, data_saved_path: str = None):
    # self.planning_log[step_num][agent_id] = {'obs':obs,'log':[(plan,counter,planning_stage,reason)], 'length': 1}
    visualize_data = convert_dict(frame_info_)
    pipeline_planning_data = convert_dict(pipeline_planning_data)
    
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
    

    with open(data_saved_path, 'w',encoding='utf-8') as json_file:
        json.dump(visualize_data, json_file, indent=4)
    print(f"Data saved to {data_saved_path}")
    return data_saved_path


def visualize_cwah_process_data_for_web(image_folder_info : dict = dict(),scene_id = 0,pipeline_planning_data :dict = None):

    concat_image_saved_folder = os.path.join(k_current_experiment_data_folder,"images",f"scene_{scene_id}")  # 替换为你的输出目录
    data_saved_path = os.path.join(k_current_experiment_data_folder, "data", f"scene{scene_id}_{VERSION}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json" )
    os.makedirs(os.path.join(k_current_experiment_data_folder, "data"), exist_ok=True)
    os.makedirs(concat_image_saved_folder, exist_ok=True)

    frame_info = concat_images(image_folder_info,output_dir= concat_image_saved_folder)
    return process_text(pipeline_planning_data = pipeline_planning_data,frame_info_ = frame_info,
                 data_saved_path = data_saved_path)




def visualize_symbolic_cwah_process_data_for_web(pipeline_planning_data :dict = None, total_step : int = 0, saving_path :str = None):
    frame_info = {}
    for i in range(total_step + 1):
        frame_info[i] = [""]

    pipeline_planning_data = convert_dict(pipeline_planning_data)
    
    agents_state = [{},{}]
    agents_latest_state = [{},{}]
    # frame_info_ key is str number not int number 
    for step_num in range(total_step + 1):
        planning_data = pipeline_planning_data.get(step_num)
        if planning_data:
            planning_data = convert_dict(planning_data)
            for agent_id in [0,1]:

                if agent_id in planning_data:
                    agent_info = planning_data[agent_id]
                    agents_state[agent_id] = agent_info
                    agents_latest_state[agent_id] = get_lastest_action_info(agent_info)
                    frame_info[step_num].append(copy.deepcopy(agent_info))
                else:
                    agent_info = agents_latest_state[agent_id]
                    frame_info[step_num].append(copy.deepcopy(agent_info))
        else:
            frame_info[step_num].extend(copy.deepcopy(agents_latest_state))
    

    with open(saving_path, 'w',encoding='utf-8') as json_file:
        json.dump(frame_info, json_file, indent=4)
    print(f"Data saved to {saving_path}")
    return saving_path







if __name__ == '__main__':

    with open(r"/media/airs/BIN/cwah_ex/cwah_planning_data/scene26_v5_20250109_143736.log", 'r',encoding='utf-8') as file:
        pipeline_planning_data = json.load(file)

    with open(r"/media/airs/BIN/cwah_ex/frame_info_from_cwah/scene_0_version_v4_None_frame_path.json", 'r',encoding='utf-8') as file:
        image_folder_info = json.load(file)
    visualize_cwah_process_data_for_web(
        image_folder_info=image_folder_info,
        scene_id=26,
        pipeline_planning_data = pipeline_planning_data
    )


