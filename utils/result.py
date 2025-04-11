import enum
import os
import json
import time

import re

def extract_to_do(input_str):
    # Input string
    # input_str = "Transport :2 calculator,1 lighter,4 iphone,1 mouse,1 purse,1 pen to the bed."

    # Extract the items and their quantities
    input_str = input_str.replace(" to the bed.", "")
    items = re.findall(r'(\d+)\s+([\w\s]+)', input_str)

    # Convert to a dictionary
    item_dict = {item[1].strip(): int(item[0]) for item in items}

    # print(item_dict)
    return item_dict


def mvp(data,visited_object):
    # print("mvp", data)
    # print()
    cnt = 0 
    for i in data:
        if i in visited_object:
            continue
        if 'target object' in i:
            cnt +=1
            visited_object.add(i)
    # print("visited_object",visited_object)
    # time.sleep(0.1)
    return cnt


def check(agent_data,visited_object):
    # jie suan
        # print("agent_id",agent_id,"agent_data['grasped_items']", agent_data['grasped_items'])
        agent_data['successful_deliveries'] += mvp(agent_data['grasped_items'],visited_object)
        agent_data['grasped_items'] = set()

def process_scene_data(data : list):

    visited_object = set()
    # 初始化机器人数据
    temp_robots = {
        0:{
        'distance_moved': 0,
        'try_grasp': 0,
        'grasped_items':  set(),
        'successful_deliveries': 0,
        'task':{},
        'action':""},
        1:{
        'distance_moved': 0,
        'try_grasp': 0,
        'grasped_items': set(),
        'successful_deliveries': 0,
        'task':{},
        'action':""}
    }
    conflict_count = 0

    ini = True
    last_conflict_action = ""
    # 处理每一帧
    for entry in data:
        # last_action = entry['actions_description']
        # last_task = entry['current_task']
        # print("entry",)
        actions = entry['actions_description'].split('.')
        current_task = entry['current_task']
        if ini:
            # print("current_task",current_task)
            ini = False
        for agent_id , action in enumerate(actions):
            parts = action.strip().split(',')
            # agent_id = int(parts[0].split(':')[1])
            action_type = parts[1].split(':')[1].strip()
            

            # 处理移动
            agent_data = temp_robots[agent_id]
            if 'go to' in action_type:
                agent_data['distance_moved'] += 0.5

                # jie suan
                check(agent_data,visited_object)
              

                if agent_data['action'] != action_type:
                    agent_data['action'] = action_type
            elif 'explore' in action_type:
                agent_data['distance_moved'] += 0.2


                # jie suan
                check(agent_data,visited_object)

                if agent_data['action'] != action_type:
                    agent_data['action'] = action_type
            # 处理抓取
            elif 'grasp' in action_type:
                agent_data['distance_moved'] += 0.3
                TEMP_FLAG = False


                # jie suan
                check(agent_data,visited_object)

                # Update the value
                if agent_data['action'] != action_type:
                    agent_data['action'] = action_type

                    if TEMP_FLAG:
                        print("grasp log", entry)

                    item = action_type.split('grasp')[-1].strip()

                    agent_data['grasped_items'].add(item)
                    agent_data['try_grasp'] += 1

                # time.sleep(1000)

            # 处理运输
            elif 'transport' in action_type:
                agent_data['distance_moved'] += 0.4

                # Update the value
                if agent_data['action'] != action_type:
                    agent_data['action'] = action_type
                    task = extract_to_do(current_task)
                    
                    if agent_data['task'] != task:
                        agent_data['task'] = task

                    # print("total_robots[agent_id]['grasped_items']", agent_data['grasped_items'])
                    # for item, count in agent_data['grasped_items']:
                    #     if item in current_task:
                    

                    # time.sleep(10000)

            elif 'put' in action_type:
                pass
            else:
                print(action_type)


        x = [action.strip().split(',')[1].split(':')[1].strip() for action in actions]
        if 'grasp' in x[0] and 'grasp' in x[1]:
            if x[0] == x[1] and x[0] != last_conflict_action:
                conflict_count += 1
                last_conflict_action = x[0]
            # agent_id = int(parts[0].split(':')[1])
            
        # # 计算交集
        # intersection = temp_robots[0]['grasped_items'].intersection(temp_robots[1]['grasped_items'])
        # conflict_count = len(intersection)
        # print("conflict_count", conflict_count)

    print(temp_robots)
    print()
    return temp_robots, conflict_count



# 设置文件夹路径
folder_path = '/media/airs/BIN/tdw_ex/training_data/try/LMs-test_rl_ablation_0.1_random/'

# 初始化总统计数据
total_robots = {}
total_conflict_count = 0
total_files = 0
load = []

# 遍历文件夹中的所有子文件夹
for i in range(0,24):
    subfolder_name = str(i)
    file_path = os.path.join(folder_path, subfolder_name, "each_frame_information.json")
    total_files += 1    
    print("file_path ",file_path)
    with open(file_path, 'r') as f:
        data = json.load(f)
    scene_data,conflict_count = process_scene_data(data)
    load.append(scene_data)

    total_conflict_count += conflict_count
    for agent_id, stats in scene_data.items():
        if agent_id not in total_robots:
            total_robots[agent_id] = {
                'distance_moved': stats['distance_moved'],
                'try_grasp': stats['try_grasp'],
                'successful_deliveries': stats['successful_deliveries'],
            }
            
        else:
            total_robots[agent_id]['distance_moved'] += stats['distance_moved']
            total_robots[agent_id]['try_grasp'] += stats['try_grasp']
            total_robots[agent_id]['successful_deliveries'] += stats['successful_deliveries']



# 计算平均值
averages = {}
for agent_id, stats in total_robots.items():
    if agent_id == 'conflict_count':
        averages['average_conflict_count'] = stats / total_files
        continue

    if agent_id not in averages:
        averages[agent_id] = {
            'average_distance_moved': 0,
            'average_try_grasp': 0,
            'average_successful_deliveries': 0,
        }
    
    averages[agent_id]['average_distance_moved'] = stats['distance_moved'] / total_files
    averages[agent_id]['average_try_grasp'] = stats['try_grasp'] / total_files
    averages[agent_id]['average_successful_deliveries'] = stats['successful_deliveries'] / total_files
    average_total_conflict_count = total_conflict_count / total_files

print("total_files", total_files)
print(averages)

print("total_conflict_count: ",total_conflict_count)
print("average_total_conflict_count: ",average_total_conflict_count)

x1 = averages[0]['average_successful_deliveries']
x2 = averages[1]['average_successful_deliveries']
x_hat = (x1 + x2)/2
junheng = ((x1 - x_hat)**2 + (x2 - x_hat)**2)**0.5
print("junheng", junheng)
# # 输出平均结果
# for agent_id, avg_stats in averages.items():
#     if agent_id == 'conflict_count':
#         print(f"  Average Conflict Count: {avg_stats:.2f}")
#         continue

#     print(f"Agent {agent_id}:")
#     print(f"  Average Distance Moved: {avg_stats['average_distance_moved']:.2f}")
#     print(f"  Average Items Carried: {avg_stats['average_try_grasp']:.2f}")
#     print(f"  Average Successful Deliveries: {avg_stats['average_successful_deliveries']:.2f}")