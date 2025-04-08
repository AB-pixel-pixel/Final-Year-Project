from framework_structure import *
task_assignment_output_rules =  """
Please think step by step. Then, generate the task assignment in the following json format:
{
  "reason": "<specific reason explanation>",
  "robot_id_task_pairs": [
    {
      "robot_id": <robot_id>,
      "action_chain": [
        "action from action space",
        "action from action space"
      ]
    },
    {
      "robot_id": <robot_id>,
      "action_chain": [
        "action from action space",
        "action from action space"
      ]
    },
  ]
}
"""

action_output_requirement = """
Please think step by step. Then, generate the action for agent to execute in the following json format:
{
    "inference_process": "The inference process of the action.",
    "reason": "The reason for choosing the action.",
    "step":  "The chosen action."
}
"""

action_chain_output_requirement = """
Please think step by step. Then, generate the action chain for agent to execute in the following json format:
{
    "inference_process": "The inference process of the action chain.",
    "intention": "The intention or reason of the action chain.",
    "action_chain": [
        "action from action space",
        "action from action space"
    ]
}
"""

refine_result_output_requirement = """
Please think step by step. Then, generate the output in the following json format:
{
    "inference_process": "The inference process of the better action chain.",
    "action_chain": [
        "action from action space",
        "action from action space"
    ]
}
"""

"""
Please generate the output in the following json format:
{
    "reason": "The reason for choosing the method. ",
    "method": "The method of refining. ",
    "action_chain": [
        "action from action space",
        "action from action space"
    ]
}
"""

conflict_solution_output_requirement = """
Based on the infomation above, decide whether to execute the current action (output the original action chain without changing) or modify the action chain.
Please think step by step. Then, generate the output in the following json format:
{
    "reason_agent0": "The reason for agent0 choosing the method. ",
    "method_agent0": "The method of agent0 refining. ",
    "action_chain0": [
        "action from action space",
        "action from action space"
    ],
    "reason_agent1" : "The reason for agent1 choosing the method. ",
    "method_agent1": "The method of agent1 refining. ",
    "action_chain1": [
        "action from action space",
        "action from action space"
    ],
}
"""
# Determine the first planned action based on your current holding state. Then, consider the state after executing the action and conceive the next action. 
cot_header = """
Think step by step, and it's recommended that the maximum planning does not exceed five steps.The distribution of items is in line with common sense; for example, fruits are more likely to be found in the kitchen. Note that a container can contain three objects, and will be lost once transported to the bed. Each robot can only put objects into the container it hold after grasping it. \
"""
cot = {'transport':"Consider 'transport' actions. Determine if you have reached the maximum limit of held items, or if you are close to the maximum step limit, in order to prevent items from being returned.",
'put':"Consider 'put' actions. Evaluate whether you need to use 'put' actions to optimize the number of items you are holding. However, the condition for performing a 'put' action is holding one container and one target object. Remember, the container can hold up to 3 objects.",
'grasp':"Consider 'grasp' actions. When your holding is not full, you should continuously plan to grasp, or pair it with some put actions, ending with a transport action. Note that grasp actions will automatically navigate to the room where the item is located, so there is no need to go to the corresponding room. Notice: Agent holds up to two objects at once. So, transport them to the bed when your hands are full.",
'explore': "Consider 'go to' and 'explore' actions."
}

thinking_cot = f"if holding nothing :\n\t{cot['grasp']}\n\t{cot['explore']}\nif holding more than one thing:\n\t{cot['transport']}\n\t{cot['put']}\n\t{cot['grasp']}\n\t{cot['explore']}"

def ensemble_cot(orders :list[str] = []):
    result_cot = []
    for order in orders:
        result_cot.append(cot[order])
    return "\n".join(result_cot)

def get_cot(observation :str):
    # 定义一个空列表，用于存储ensemble_order
    ensemble_order = []
    # 如果observation中包含"holding a"，则将ensemble_order设置为['grasp','explore','transport','put']
    if "holding a container" in observation and "and a target object" in observation:
        ensemble_order = ['put','transport']
    if "holding a" in observation:
        ensemble_order = ['grasp','explore','transport','put']
    # 如果observation中包含"holding nothing"，则将ensemble_order设置为['grasp','explore']
    elif "holding nothing" in observation:
        ensemble_order = ['grasp','explore']    
    elif "holding two" in observation:
        ensemble_order = ['transport']
    else:
        ensemble_order = ['grasp','explore','transport']
    # 调用ensemble_cot函数，将ensemble_order作为参数传入，并将返回值赋值给temp_cot
    temp_cot = ensemble_cot(ensemble_order)
    # 将temp_cot和提示信息拼接成字符串，赋值给ans
    ans = f"System strongly recommend consider the following action: {temp_cot}\n"
    # 返回ans
    return ans 

gpt_4o_mini_rule1= "When selecting actions from the action space, abandon the alphabetical index and directly copy the action content, preserving all letters."


# good_pattern = # "Good pattern is : If there are target items available to grasp, prioritize grasping them and then transport it. The best pattern is explore -> go grasp ... -> grasp ... -> transport objects I'm holding to the bed. -> explore -> grasp ... -> grasp ... -> transport objects I'm holding to the bed."



action_format_constrains = "Each step only can represent a minimal action, which follow a specific format, such as: go to <room_name>; put <object_name> into the container <container_name>; go grasp target object <object_name>; explore current room <room_name>; transport objects I'm holding to the bed. "


def get_task_assignment_output_rules(text : str):
    temp_cot = get_cot(text)
    return f"""For each identified action:
    - Refer to objects explicitly by their name and ID (e.g., <table> (712)). 
    - Actions should be selected from action space.
    - In the plan, when agent enters a new room, follow a the command 'replan' as action. This command will call replan to handle new discovery in the room.
    - If you don't know where is the bed, just try to go to bedroom. The 'transport' can only be executed when found the bed. {cot_header + temp_cot}\n
    {gpt_4o_mini_rule1}""" + task_assignment_output_rules


def get_refining_rules(text: str):
    temp_cot = get_cot(text)
    return f"""
        **We expect you:**  
        - Fix minor issues (e.g., typos, alignment) while keeping the original intent.
        - Ensure:
            - All rooms/objects use explicit names + IDs.  
            - Actions should select from action space.
            {gpt_4o_mini_rule1}
            - {temp_cot}
        {refine_result_output_requirement}
        """ 


def get_create_action_chain_rules(text : str):
    # 获取当前文本的COT
    temp_cot = get_cot(text)
    # 返回创建动作链规则
    return f"""Based on the information, \
    Please make decision about wheather grasp the target object that in your current room or transport it to the bed.\
    Each robot can hold two things at a time, and they can be objects or containers. Each robot can grasp containers and put objects into them to hold more objects at a time. \
    1. The first step should be suitable for the actions can be performed. \
    2. {temp_cot} \
    3. If you don't know where is the bed, just try to go to bedroom. The 'transport' can only be executed when found the bed. \
    {gpt_4o_mini_rule1}
    {action_chain_output_requirement}"""

def get_refine_action_step_rules(text: str):
    # 获取cot
    temp_cot = get_cot(text)
    # 返回一个字符串，包含修改步骤的规则
    return f"Based on the information, \
    Please make modifications to the step you are failled in while maintaining its overall intention.\
    The new step you create will replace the failed step, so: \
    1. The new step should be suitable for the actions can be performed.\n \
    2. {temp_cot}\n\
    3. If you don't know where is the bed, just try to go to bedroom. The 'transport' can only be executed when found the bed.\n \
    You should specifically point out the names of the rooms to visit and their IDs, to avoid confusing rooms with the same name.\n \
    Each step only can represent a minimal action, which follow a specific format, such as: go to <room_name>; put <object_name> into the container <container_name>; go grasp target object <object_name>; explore current room <room_name>; transport objects I'm holding to the bed.\n{gpt_4o_mini_rule1}\n{action_output_requirement}"


def get_action_chain_mutation_rules(text: str):
    # 获取cot
    temp_cot = get_cot(text)
    # 返回一个字符串，包含修改步骤的规则
    return f"When a step fails, create a replacement step that: \n 1. Clearly identifies any room, container, or object with its name and ID.\n2. Follows the minimal action format (e.g., 'go to explore <room> (id)').\n3. {temp_cot}.\n{gpt_4o_mini_rule1}\n{action_chain_output_requirement}"


def get_solve_conflict_rules(text: str):
    # 获取cot
    temp_cot = get_cot(text)
    # 返回一个字符串，包含解决冲突的规则
    return f"{temp_cot}\n{gpt_4o_mini_rule1}\n{conflict_solution_output_requirement}"

