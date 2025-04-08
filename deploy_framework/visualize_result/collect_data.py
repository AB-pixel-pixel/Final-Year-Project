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


def collect_action_history(pipeline_planning_data_paths :dict = None):
    # load data
    agent_state_data = []

    for agent_id, current_step_data_path in pipeline_planning_data_paths.items():


        with open(current_step_data_path, 'r',encoding='utf-8') as json_file:
            # {"step_num": self.public_pipeline_state["step_num"],
            # "target_step":target_step,
            # "chosen_action": option})
            current_step_data = json.load(json_file)
        
        text = ",".join([str(steps['target_step']) for steps in current_step_data])
        print(text)
    
    #print(agent_state_data)
    # return agent_state_data


if __name__ == '__main__':
    # for scene  in range(24):

    planning_log_dir = "/home/airs/bin/ai2thor_env/cb/planning_data"
    paths = {0:"/home/airs/bin/ai2thor_env/cb/planning_data/scene18_current_step_data_v12_0_20241217_173419.json",
            1:"/home/airs/bin/ai2thor_env/cb/planning_data/scene18_current_step_data_v12_1_20241217_173419.json"}

    collect_action_history(pipeline_planning_data_paths=paths)
