import os
import csv
import numpy as np
import json
import re
import time
from sympy import evaluate
from communication_protocol import AIRS_LLM
from tqdm import tqdm

GLOBAL_LLM = AIRS_LLM(large_model_server_ip_port = "http://127.0.0.1:8040/api/generate") # "http://192.168.31.81:2048/api/generate")# 


class RewardRewriteSystem:
    def __init__(self, context_window_size=5):
        if context_window_size % 2 == 0:
            raise ValueError("上下文窗口大小必须为奇数")
        self.context_window_size = context_window_size  # 上下文窗口长度
        self.half_window = context_window_size // 2
        self.debug = False


    def generate_module_a_template(self, current_index, episode_experiences):
        """
        模块A：生成提示词模板，分析任务变化、观测情况、动作空间变化
        :param current_index: 当前经验在episode中的索引
        :param episode_experiences: 整个episode的经验列表，每个元素是一个七元组
        :return: 模块A的提示词模板字符串
        """
        try:
            # 获取当前经验
            current_experience = episode_experiences[current_index]            
            evaluate_exp = f"Currently we are going to analyze the target action."


            # 计算上下文窗口范围
            start_index = max(0, current_index - self.half_window)
            end_index = min(len(episode_experiences), current_index + self.half_window + 1)
            
            # 提取上下文经验
            context_experiences = episode_experiences[start_index:end_index]
            
            # 构建上下文部分
            context_info = "Contextual Experience:\n"
            for exp_id, exp in enumerate(context_experiences):
                frame, aid, obs, tsk, act, act_space, rew = exp
                # print("type(act_space): ",type(act_space))
                if exp_id < self.half_window:
                    context_info += f"The previous interaction info: frame {frame}, Agent {aid} performs action {act} based observation {obs}, with length of action space {len(act_space)} and goal finish situation {tsk}.\n"
                elif exp_id == self.half_window:
                    context_info += f"The interaction info which you need to evaluate, also called target action: frame {frame}, Agent {aid} performs action {act} based observation {obs} and goal finish situation {tsk} , with length of action space {len(act_space)}.\n"
                else:
                    context_info += f"The interaction info after the action executed: frame {frame}, Agent {aid} performs action {act} based observation {obs} with length of action space {len(act_space)} and goal finish situation {tsk}.\n"


            prompt = evaluate_exp
            
            prompt += "Please analyze the changes before and after the target action in the following four aspects and provide a score (0 or 1) and reasoning:\n"
            prompt += "1. Task completion effectiveness: Evaluate the contribution of the target action to task completion. If the target action is relative to the task give 1 score. Otherwise, give 0 score. \n"
            prompt += "2. Increament of action space: If the target action increase the action space after being executed, give 1 score, otherwise, give 0 score.\n"
            prompt += "3. Observation coverage increament: Evaluate the coverage of the environment observations increments created by the target action. If the coverage increases, give 1 score. Otherwise, give 0 score.\n"
            prompt += "4. Robot action collaboration: Evaluate the collaboration of the target action with other agents' actions. If they are doing the same action, give 0 score. Otherwise, give 1 score.\n"
            prompt += "Please output the result in JSON format, structured as follows:\n"
            prompt += "{\n"
            prompt += "  \"Inference Part\": \"...\",\n"
            prompt += "  \"Score Section\": {\n"
            prompt += "    \"Task completion effectiveness\": X,\n"
            prompt += "    \"Increament of action space\": Y,\n"
            prompt += "    \"Observation Coverage Increament\": Z,\n"
            prompt += "    \"Robot Action Collaboration\": W\n"
            prompt += "  },\n"
            prompt += "  \"Reason for Scoring\": \"...\"\n"
            prompt += "}"
            # 组合输入模板
            module_a_template = context_info + "\n" + prompt # + current_info + "\n" 
            
            return module_a_template
        
        except Exception as e:
            print(f"生成模块A提示词模板时出错：{e}")
            return None
    
    def generate_module_b_template(self, current_index, episode_experiences, evaluation):
        """
        模块B：生成提示词模板，根据模块A的评分改写奖励值
        :param current_index: 当前经验在episode中的索引
        :param episode_experiences: 整个episode的经验列表，每个元素是一个七元组
        :param evaluation: 模块A生成的评分和理由
        :return: 模块B的提示词模板字符串
        """
        try:
            # 计算上下文窗口范围
            start_index = max(0, current_index - self.half_window)
            end_index = min(len(episode_experiences), current_index + self.half_window + 1)
            
            
            # 提取上下文经验
            context_experiences = episode_experiences[start_index:end_index]
            
            # 构建上下文部分
            context_info = "Contextual Experience:\n"
            for exp_id, exp in enumerate(context_experiences):
                frame, aid, obs, tsk, act, act_space, rew = exp
                # print("type(act_space): ",type(act_space))
                if exp_id < self.half_window:
                    context_info += f"The previous interaction info: frame {frame}, Agent {aid} performs action {act} based observation {obs}, with length of action space {len(act_space)} and goal finish situation {tsk}.\n"
                elif exp_id == self.half_window:
                    context_info += f"The interaction info which you need to evaluate, also called target action: frame {frame}, Agent {aid} performs action {act} based observation {obs} and goal finish situation {tsk} , with length of action space {len(act_space)}.\n"
                else:
                    context_info += f"The interaction info after the action executed: frame {frame}, Agent {aid} performs action {act} based observation {obs} with length of action space {len(act_space)} and goal finish situation {tsk}.\n"

                
            

            # 构建提示词部分
            prompt = "Please update the reward values based on the information above. The rewriting criteria are as follows:\n"
            prompt += f"Evaluation from Module A:\n{evaluation}\n"
            prompt += "If this action greatly promotes the completion of the task, the reward value is 3. If it does not help with task completion, the reward value is -3. If it is neither helpful nor unhelpful, the reward value is 0."
            prompt += "the reward value should be in the range of [-3 , 3].\n"
            prompt += "Please output the result in JSON format, structured as follows:\n"
            prompt += "{\n"
            prompt += "  \"Inference Part\": \"...\",\n"
            prompt += "  \"Rewritten Reward Value\": X,\n"
            prompt += "  \"Reason for Rewriting\": \"...\"\n"
            prompt += "}"
            
            # 组合输入模板
            module_b_template =  context_info + "\n"  + prompt
            
            return module_b_template
        
        except Exception as e:
            print(f"生成模块B提示词模板时出错：{e}")
            return None
    
    def wrap_llm_response(self, prompt, is_module_a=True):
        """
        包装大语言模型的响应，提取JSON格式的结果
        :param prompt: 输入提示词
        :param is_module_a: 是否是模块A的请求
        :return: 提取后的JSON结果或None
        """
        try:
            # 调用大语言模型接口
            response = GLOBAL_LLM.invoke(prompt)
            ans = response['response']
            
            # 提取最后一对{}的内容
            json_str = re.findall(r'{[^{}]*}', ans)[-1]
            json_str = json_str.strip()
            
            # 将JSON字符串转换为字典
            result = json.loads(json_str)
            
            return result
        
        except Exception as e:
            print(f"包装大语言模型响应时出错：{e}")
            return None
    
    def rewrite_rewards(self, episode_experiences):
        """
        对整个episode的经验进行奖励值改写
        :param episode_experiences: 整个episode的经验列表，每个元素是一个七元组
        :return: 改写奖励值后的episode经验列表
        """
        try:
            # 遍历每个经验，生成模块A和模块B的提示词模板并改写奖励值
            # for i in range(len(episode_experiences)):
            for i in tqdm(range(len(episode_experiences)), desc="处理经历"):
                if self.debug:
                    print(f"处理帧{episode_experiences[i][0]}，当前上下文窗口大小：{self.context_window_size}")
                
                max_try_times = 2
                try_cnt = 0
                evaluation = None
                while evaluation is None:
                    # 生成模块A的提示词模板
                    module_a_template = self.generate_module_a_template(i, episode_experiences)
                    if self.debug:
                        print(f"###################模块A的提示词模板：###################\n{module_a_template}")
                    
                    # 获取模块A的评分和理由
                    module_a_result = self.wrap_llm_response(module_a_template, is_module_a=True)
                    print("module_a_result:",module_a_result)
                    if module_a_result:
                        if self.debug:
                            print(f"###################模块A的输出：###################\n{json.dumps(module_a_result, indent=2, ensure_ascii=False)}")
                        evaluation = json.dumps(module_a_result, ensure_ascii=False)
                    else:
                        print("###################模块A的输出提取失败，重新规划...###################")

                    try_cnt +=1
                    if try_cnt > max_try_times:
                        evaluation = "Evaluation failed."
                        break
                
                # 生成模块B的提示词模板
                module_b_template = self.generate_module_b_template(i, episode_experiences, evaluation)
                if self.debug:
                    print(f"###################模块B的提示词模板：###################\n{module_b_template}")
                
                # 获取模块B的改写后的奖励值
                max_try_times = 2
                try_cnt = 0
                new_reward = None
                while new_reward is None:
                    module_b_result = self.wrap_llm_response(module_b_template, is_module_a=False)
                    if module_b_result:
                        if self.debug:
                            print(f"###################模块B的输出：###################\n{json.dumps(module_b_result, indent=2, ensure_ascii=False)}")
                        new_reward = module_b_result.get("Rewritten Reward Value", 0)
                        new_reward = max(-3,new_reward)
                        new_reward = min(3,new_reward)

                    else:
                        print("###################模块B的输出提取失败，重新规划...###################")
                    try_cnt +=1
                    if try_cnt > max_try_times:
                        new_reward = 0
                    print("new_reward:", new_reward)

                # 更新奖励值
                episode_experiences[i] = episode_experiences[i][:6] + (new_reward,)
                if self.debug:
                    print(f"###################帧{episode_experiences[i][0]}的奖励值已更新为：{new_reward}###################")
            
            return episode_experiences
        
        except Exception as e:
            print(f"奖励值改写时出错：{e}")
            return episode_experiences

def read_scene_data(scene_folder):
    """
    读取scene_x文件夹中的train_data.csv数据
    :param scene_folder: scene_x文件夹路径
    :return: 包含该scene所有经验的列表
    """
    scene_experiences = []
    csv_file = os.path.join(scene_folder, "train_data.csv")
    if os.path.exists(csv_file):
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                # 解析每一行数据
                frame = int(row[0])
                agent_id = int(row[1])
                observation = row[2]
                task = row[3]
                action = row[4]
                action_space = eval(row[5])  # 将动作空间从字符串转换为列表
                reward = int(row[6].split('.')[0])
                # 将数据作为元组添加到场景经验列表中
                scene_experiences.append((frame, agent_id, observation, task, action, action_space, reward))
    return scene_experiences

import pickle
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 示例用法
if __name__ == "__main__":
    start_time = time.time()
    # 假设scene文件夹路径
    scenes_folder = "/media/airs/BIN/graduation_design_data_training"  # 替换为实际的scene文件夹路径
    # 遍历所有scene_x文件夹
    scene_list = [d for d in os.listdir(scenes_folder) if os.path.isdir(os.path.join(scenes_folder, d)) and d.startswith("scene_")]
    all_scenes_data = []  # 存储所有scene的数据
    all_scenes_data_len = 0
    for scene in scene_list:
        scene_id = int(scene.split('_')[1])
        if scene_id != 2 and scene_id != 1 and scene_id != 3:
            continue
        scene_path = os.path.join(scenes_folder, scene)
        scene_data = read_scene_data(scene_path)
        if scene_data:
            all_scenes_data.append(scene_data)
            all_scenes_data_len += len(scene_data)
            print(f"成功读取{scene}的数据，包含{len(scene_data)}条经验")
    print(f"data has {all_scenes_data_len} items.")

    # 创建奖励改写系统，上下文窗口大小为5（奇数）
    reward_rewriter = RewardRewriteSystem(context_window_size=3)
    
    # 对每个scene的经验进行奖励值改写
    for i, scene_data in enumerate(all_scenes_data):
        print(f"开始处理scene_{i+1}的数据...")
        rewritten_scene_data = reward_rewriter.rewrite_rewards(scene_data)
        print(f"scene_{i+1}的数据处理完成")
        # 可以将改写后的数据保存或进一步处理
        all_scenes_data[i] = rewritten_scene_data
    
    # 输出改写后的结果示例
    # print("改写后的scene_1经验：")
    # for exp in all_scenes_data[0]:
    #     print(exp)

    # 保存数据到pickle文件
    with open('./all_scenes_data_v3.pkl', 'wb') as f:
        pickle.dump(all_scenes_data, f)

    print(f"数据保存完成，耗时：{time.time() - start_time}秒")