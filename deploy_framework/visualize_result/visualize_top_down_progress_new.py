import cv2
import os
import re
import numpy as np
import time
import textwrap
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import k_ex_result_folder

import json


def visualize_progress(image_folder,_index = 0,save_folder = "",
        scene_name = "-1", version = "-1",pipeline_planning_data_paths :dict = None,
        each_frame_information_path = "",success_number:int = 0):

    os.makedirs(save_folder,exist_ok=True)

    # load data

    # self.each_frame_information.append(
    # {"num_frames": num_frames, 
    # "actions_description": actions_description, 
    # "current_task": current_task,"image_path": path})

    with open(each_frame_information_path, 'r') as json_file:
        each_frame_information_data = json.load(json_file)



    agent_state_data = []

    for agent_id, steps_data_current_step_data_path in pipeline_planning_data_paths.items():
        steps_data_path , current_step_data_path = steps_data_current_step_data_path
        general_data = dict()
        with open(steps_data_path, 'r',encoding='utf-8') as json_file:
            # _agent.steps_data.append(
            # {"step_num":self.public_pipeline_state["step_num"],
            # "cot_counter": _agent.state['cot_counter'],
            # "steps":" | ".join([step.step for step in cot.chain])})
            steps_data = json.load(json_file)
        #   agent_steps_data.append(steps_data)
            # print("steps_data",steps_data)
        for steps in steps_data:
            text = f"{steps['steps']}"
            general_data[steps['step_num']] = {"plan":text}

            # if general_data.get(steps['step_num']):
            #     general_data[steps['step_num']] = general_data[steps['step_num']]['plan'] + "\n" + text
            # else:
        with open(current_step_data_path, 'r',encoding='utf-8') as json_file:
            # {"step_num": self.public_pipeline_state["step_num"],
            # "target_step":target_step,
            # "chosen_action": option,
            # "available_plans:": "A. [goexplore] <bathroom> (11)\nB. [gocheck] <kitchencabinet> (133)\n"})
            current_step_data = json.load(json_file)
        
        for steps in current_step_data:
            text = f"target_step:{steps['target_step']},chosen_action:{steps['chosen_action']}"
            if general_data.get(steps['step_num']):
                general_data[steps['step_num']].update({"state:":text})
            else:
                general_data[steps['step_num']] = {"state:":text}

        agent_state_data.append(general_data)

    # for key in general_data1:
    #     if key in general_data0:
    #         general_data0[key] += general_data1[key]  # 合并相同键的值
    #     else:
    #         general_data0[key] = general_data1[key]
    # all_keys = sorted(general_data0.keys())



    _initial_flag = True

    split_num = len(agent_state_data)
    previous_values = {i:dict() for i in range(split_num)}
    # align data
    for frame_info in each_frame_information_data:
        num_frames = frame_info["num_frames"]
        for index,state_data in enumerate(agent_state_data):
            if num_frames in state_data:
                # print("previous_values[index]",previous_values[index])
                # print("state_data[num_frames]",state_data[num_frames])
                previous_values[index].update(state_data[num_frames])
                # print("previous_values[index]",previous_values[index])
        # print("previous_values\n",previous_values)
        # actions_description = frame_info["actions_description"]
        current_task = frame_info["current_task"]
        text = []
        text.append(f"num_frames:{num_frames}")
        text.append(current_task)
        for key, value in previous_values.items():
            for key1, value1 in value.items():
                text.append(value1) # 执行两次
        # print("text\n",text)
        
        image_path = frame_info["image_path"]
        # print("读取俯视图： ",image_path)
        image = cv2.imread(image_path)
        # 获取原图像的尺寸
        map_size = image.shape[:2]  # (高度, 宽度)
        # 仅在底部增加白边
        border_height = 700  # 白边的高度
        bordered_map = np.zeros((map_size[0], map_size[1] + border_height, 3), dtype=np.uint8)
        bordered_map[:, :map_size[1]] = image  # 将原图像放置在新图像的顶部
        bordered_map[:, map_size[1]:] = [255, 255, 255]  # 设置底部边框为白色
        standard_height = map_size[0]//split_num
        # 添加字幕
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # 字体缩放比例，调整为更小的值
        thickness = 1  # 字体厚度

        # draw topic
        # print(f"draw topic: {topic}")
        counter = 0
        level_counter = 0
        # chunk_text = str(text).replace("，",",").split(",")
        # first draw topic , then draw the agent state
        for chunk_text in text:
            lines = textwrap.wrap(chunk_text, width=100)
            for line in lines:
                if level_counter  <= 1:
                    cv2.putText(bordered_map, line, (map_size[1], 10 + 15 * counter), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                elif level_counter <= 3:
                    cv2.putText(bordered_map, line, (map_size[1], 30 + 15 * counter), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                else:
                    cv2.putText(bordered_map, line, (map_size[1], 70 + 15 * counter), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                counter += 1
            level_counter += 1
        # # draw line
        # for agent_id , agent_action in line.items():
        #     line = f"agent_id:{agent_id},action:{agent_action}"
        #     # print(line)
        #     cv2.putText(bordered_map, line, (10, 2*map_size[0]//3 + 40 * counter), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        #     counter += 1

        if _initial_flag:
            # 读取第一张图片获取尺寸
            height, width, layers = bordered_map.shape
            # 定义视频编码器和输出文件
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编码
            # video_path = os.path.join(image_folder, "output.mp4")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(save_folder, f"scene_{scene_name}_version_{version}_{timestamp}_success_{success_number}_20fps.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height)) # 假设帧率为30fps
            print(f"ready to write in {video_path}")
        _initial_flag = False
        video_writer.write(bordered_map)
        

    video_writer.release()
    print(f"视频已保存到: {video_path}")





  

if __name__ == '__main__':
    # for scene  in range(24):
    scene = 20
    scene = str(scene)
    each_frame_information_path = os.path.join(k_ex_result_folder, scene, 'each_frame_information.json')
    image_folder = f"{k_ex_result_folder}/{scene}/top_down_image"
    # each_frame_information_path = f"{k_ex_result_folder}/{scene}/each_frame_information.json"
    print("image folder:",image_folder)

    planning_log_dir = "/home/airs/bin/ai2thor_env/cb/planning_data"
    paths = {0:("/home/airs/bin/ai2thor_env/cb/planning_data/scene20_steps_data_v10_0_20241215_015023.json",
            "/home/airs/bin/ai2thor_env/cb/planning_data/scene20_current_step_data_v10_0_20241215_015023.json"),
            1:("/home/airs/bin/ai2thor_env/cb/planning_data/scene20_steps_data_v10_1_20241215_015023.json",
               "/home/airs/bin/ai2thor_env/cb/planning_data/scene20_current_step_data_v10_1_20241215_015023.json")}

    if os.path.exists(image_folder):
        visualize_progress(image_folder,save_folder="/media/airs/BIN/top_down_result123",
            scene_name = scene, each_frame_information_path = each_frame_information_path,
            pipeline_planning_data_paths=paths )
            