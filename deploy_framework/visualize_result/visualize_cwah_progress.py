import cv2
import os
import re
import numpy as np
import time
import textwrap
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import k_ex_result_folder,k_current_experiment_data_folder

import json

import os
from PIL import Image

def visualize_progress(save_folder = "",
        scene_name = "-1", version = "-1",pipeline_planning_data_paths :dict = None,
        each_frame_information_data:dict = {},success_number:int = 0,current_tasks_path = ""):

    os.makedirs(save_folder,exist_ok=True)

    # load data


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

    with open(current_tasks_path, 'r',encoding='utf-8') as json_file:
        # {"step_num": self.public_pipeline_state["step_num"],
        # "current_task":self.public_pipeline_state['current_task']}
        current_tasks_raw_data = json.load(json_file)
        current_tasks_data = {}
        for item in current_tasks_raw_data:
            current_tasks_data[item['step_num']] = item['current_task']


    _initial_flag = True

    split_num = len(agent_state_data)
    previous_values = {i:dict() for i in range(split_num)}

    # align data

    # self.each_frame_information.append(
    # self.video_info_path[self.steps] = []
    # append(str(agent_id))
    # append(temp_image_path)
    current_task_text = ""
    for num_frames, frame_info in each_frame_information_data.items():
        
        for index,state_data in enumerate(agent_state_data):
            if num_frames in state_data:
                previous_values[index].update(state_data[num_frames])

        if num_frames in current_tasks_data:
            current_task_text = current_tasks_data[num_frames]

        text = []
        text.append(f"num_frames:{num_frames}")
        text.append(current_task_text)
        for key, value in previous_values.items():
            for key1, value1 in value.items():
                text.append(value1) # 执行两次
        # print("text\n",text)
        
        image_path = frame_info["image_path"]
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
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            fps_rate = 1
            video_path = os.path.join(save_folder, f"scene_{scene_name}_version_{version}_{timestamp}_success_{success_number}_{fps_rate}fps.mp4")
            video_writer = cv2.VideoWriter(video_path, fourcc, fps_rate ,(width, height)) # 假设帧率为30fps
            print(f"ready to write in {video_path}")
        _initial_flag = False
        video_writer.write(bordered_map)
        

    video_writer.release()
    print(f"视频已保存到: {video_path}")





def concat_images(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有图像文件
    images = [f for f in os.listdir(input_dir) if f.endswith('_rgb.png')]
    
    # 提取帧数
    frame_numbers = set(int(f.split('_')[0]) for f in images)

    global_info = {}
    # 处理图像
    for frame in sorted(frame_numbers):  # 动态处理帧数
        left_image_path = os.path.join(input_dir, f"{frame}_0_rgb.png")
        right_image_path = os.path.join(input_dir, f"{frame}_1_rgb.png")
        
        # 检查左右图像是否存在
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
            # global_info[frame] = {"image_path": f"{frame}.png"}
            global_info[frame] = {"image_path": os.path.abspath(new_image_path)}
    return global_info



def process_text(
    scene_name = "-1", version = "-1",pipeline_planning_data_paths :dict = None,success_number:int = 0,
    global_info:dict = {}):


    # load data
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
    general_data_path = f"/home/airs/bin/ai2thor_env/cb/cwah_planning_data/scene_{scene_name}_version_{version}_general_data_path.json"
    with open(general_data_path, 'w',encoding='utf-8') as json_file:
        json.dump(agent_state_data, json_file, indent=4)
    # for key in general_data1:
    #     if key in general_data0:
    #         general_data0[key] += general_data1[key]  # 合并相同键的值
    #     else:
    #         general_data0[key] = general_data1[key]
    # all_keys = sorted(general_data0.keys())



    _initial_flag = True

    split_num = len(agent_state_data)
    previous_values = {i:dict() for i in range(split_num)}
   
    for num_frames, info in global_info.items():
        image_path0, image_path1 = info
        
        for index,state_data in enumerate(agent_state_data):
            if num_frames in state_data:
                previous_values[index].update(state_data[num_frames])

        text = []

        left_text = [f"num_frames:{num_frames}"]
        right_text = [f"num_frames:{num_frames}"]

        for key, value in previous_values.items():
            for key1, value1 in value.items():
                text.append(value1) # 执行两次
        # print("text\n",text)
        
        image_path = frame_info["image_path"]
        # print("读取俯视图： ",image_path) 

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
if __name__ == '__main__':
    scene_id =  0
    input_path = f'/media/airs/BIN/cwah/cwah/test_results/data_collection_vision_LLMs_framework_test_v4_scene{scene_id}'
    output_path = f'/home/airs/bin/ai2thor_env/concat_image_ouput/scene_{scene_id}'  # 替换为你的输出目录
    global_info = concat_images(input_path, output_path)
    # print(global_info)


    current_tasks_path = f"/home/airs/bin/ai2thor_env/cb/cwah_planning_data/scene_{scene_id}_v4_current_task.json"

    paths = {0: (f'/home/airs/bin/ai2thor_env/cb/planning_data/scene{scene_id}_steps_data_v4_0_20250103_131329.json', 
                 f'/home/airs/bin/ai2thor_env/cb/planning_data/scene{scene_id}_current_step_data_v4_0_20250103_131329.json'), 
             1: ('/home/airs/bin/ai2thor_env/cb/planning_data/scene26_steps_data_v4_1_20250103_131329.json', 
                 '/home/airs/bin/ai2thor_env/cb/planning_data/scene26_current_step_data_v4_1_20250103_131329.json')}
    # {0: ('/home/airs/bin/ai2thor_env/cb/planning_data/scene5_steps_data_v4_0_20250102_192351.json', '/home/airs/bin/ai2thor_env/cb/planning_data/scene5_current_step_data_v4_0_20250102_192351.json'), 1: ('/home/airs/bin/ai2thor_env/cb/planning_data/scene5_steps_data_v4_1_20250102_192351.json', '/home/airs/bin/ai2thor_env/cb/planning_data/scene5_current_step_data_v4_1_20250102_192351.json')}
    
    visualize_progress(save_folder="/home/airs/bin/ai2thor_env/cb/cwah_images/video",
                       scene_name=str(scene_id),version="v4",
                       pipeline_planning_data_paths=paths,each_frame_information_data=global_info,
                       current_tasks_path=current_tasks_path)
    
    # planning_log_dir = "/home/airs/bin/ai2thor_env/cb/planning_data"
    # {0:("/home/airs/bin/ai2thor_env/cb/planning_data/scene20_steps_data_v10_0_20241215_015023.json",
    #         "/home/airs/bin/ai2thor_env/cb/planning_data/scene20_current_step_data_v10_0_20241215_015023.json"),
    #         1:("/home/airs/bin/ai2thor_env/cb/planning_data/scene20_steps_data_v10_1_20241215_015023.json",
    #            "/home/airs/bin/ai2thor_env/cb/planning_data/scene20_current_step_data_v10_1_20241215_015023.json")}

    # if os.path.exists(image_folder):
    #     visualize_progress(image_folder,save_folder="/media/airs/BIN/top_down_result123",
    #         scene_name = scene, each_frame_information_path = each_frame_information_path,
    #         pipeline_planning_data_paths=paths )
            