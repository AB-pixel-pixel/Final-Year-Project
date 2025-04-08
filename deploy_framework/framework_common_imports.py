

from framework_structure import *




# 软标准

def evaluate_observation_changed_a_lot(current_obs:str, frozen_obs:str) -> bool:
    current_obs_num = current_obs.count("found")
    frozen_obs_num = frozen_obs.count("found")
    if current_obs_num == frozen_obs_num:
        # 如果前后观测到的物体数量差不多，继续按计划进行
        return False
    elif current_obs_num > frozen_obs_num + 2:
        # 如果前后观测到的物体数量相差较大，建议重新规划一次
        return True
    else:
        return False


def simplify_():
    pass


import json


def extract_last_json(text):
    json_list = []
    length = len(text)
    in_json = False
    start = 0
    stack = 0

    for i in range(length):
        if text[i] == '{':
            if not in_json:
                in_json = True
                start = i
            stack += 1
        elif text[i] == '}':
            stack -= 1
            if stack == 0:
                json_str = text[start:i+1]
                try:
                    data = json.loads(json_str)
                    json_list.append(data)
                except json.JSONDecodeError:
                    # 如果解析失败，跳过这个潜在的json
                    pass
                in_json = False

    if json_list:
        return json_list
    else:
        return []