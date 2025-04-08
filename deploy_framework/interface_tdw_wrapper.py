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

cot_header = """
Determine the first planned action based on your current holding state. Then, consider the state after executing the action and conceive the next action. Think step by step, and it's recommended that the maximum planning does not exceed five steps."""
cot = {'transport':"'transport' actions. Determine if you have reached the maximum limit of held items, or if you are close to the maximum step limit, in order to prevent items from being returned.",
'put':"'put' actions. Evaluate whether you need to use 'put' actions to optimize the number of items you are holding. However, the condition for performing a 'put' action is holding one container and one target object. Remember, the container can hold up to 3 objects.",
'grasp':"'grasp' actions. When your holding is not full, you should continuously plan to grasp, or pair it with some put actions, ending with a transport action. Note that grasp actions will automatically navigate to the room where the item is located, so there is no need to go to the corresponding room. Notice: Agent holds ≤2 objects at once",
'explore': "when there are no more actions available, consider 'go to' and 'explore' actions."
}

thinking_cot = f"if holding nothing :\n\t{cot['grasp']}\n\t{cot['explore']}\nif holding more than one thing:\n\t{cot['transport']}\n\t{cot['put']}\n\t{cot['grasp']}\n\t{cot['explore']}"

def ensemble_cot(orders :list[str] = []):
    result_cot = []
    for order in orders:
        result_cot.append(cot[order])
    return "\n".join(result_cot)

def get_cot(observation :str):
    ensemble_order = []
    if "holding a" in observation:
        ensemble_order = ['grasp','explore','transport','put']
    elif "holding nothing" in observation:
        ensemble_order = ['grasp','explore']    
    temp_cot = ensemble_cot(ensemble_order)
    ans = f"System strongly recommend consider the following action: {temp_cot}\n"
    return ans 

gpt_4o_mini_rule1= "Each action should come from the action space, abandon the alphabetical index and directly copy the action content, preserving all letters."


# good_pattern = # "Good pattern is : If there are target items available to grasp, prioritize grasping them and then transport it. The best pattern is explore -> go grasp ... -> grasp ... -> transport objects I'm holding to the bed. -> explore -> grasp ... -> grasp ... -> transport objects I'm holding to the bed."


explanation_game = "When you need to drop the holding target objects which means holding two objects in your hands, just select the 'transport' action. \ngo grasp type action will automatically navigate to the room, no need to go to the room. When holding one target object and one container in hands, plan put xxx into the container to free one hand. One container can contain up to three objects."

action_format_constrains = "Each step only can represent a minimal action, which follow a specific format, such as: go to <room_name>; put <object_name> into the container <container_name>; go grasp target object <object_name>; explore current room <room_name>; transport objects I'm holding to the bed. "


# def get_task_assignment_output_rules(text : str):
#     temp_cot = get_cot(text)
#     return f"""For each identified action:
#     - Agents can hold up to two objects; refer to objects explicitly by their name and ID (e.g., <table> (712)). 
#     - If you don't know where is the bed, just try to go to bedroom. The 'transport' can only be executed when found the bed. {cot_header + temp_cot}\n
#     {gpt_4o_mini_rule1}""" + task_assignment_output_rules

# #  - In the plan, when agent enters a new room, follow a the command 'replan' as action. This command will call replan to handle new discovery in the room.


# def get_refining_rules(text: str):
#     temp_cot = get_cot(text)
#     return f"""
#         **We expect you:**  
#         - Fix minor issues (e.g., typos, alignment) while keeping the original intent.  
#         - Ensure:
#             - All rooms/objects use explicit names + IDs.  
#             - Actions follow minimal syntax. 
#             {gpt_4o_mini_rule1}
#             - {temp_cot}
#         {refine_result_output_requirement}
#         """ 
def get_initial_public_pipeline() -> dict:
    # batch_prompt_elements = batch_prompt_elements.batch_prompt_constituent_elements

    PIPELINE_STATE = dict()

    # =============================================================================
    # Task assignment
    # =============================================================================
    PIPELINE_STATE["task_assignment_role"] = """You are an expert in task assignment for robotic agents. """
        
    PIPELINE_STATE["task_assignment_example"] = ""
    
    PIPELINE_STATE['task_assignment_output_rules'] = ""
    #     - Briefly consider its cost and benefit internally as you plan.

    # =============================================================================
    # Create new action chain 
    # =============================================================================
    PIPELINE_STATE["create_action_chain_role"] = "You are an expert in creating servel steps of planning."

    PIPELINE_STATE["create_action_chain_examples"] = "" 
    _ = """This two examples from another scene illustrates an effective pattern that can be referenced for the task allocation steps, but the target object may not consist with ours target: \
    Example1: Instruction: Transport 1 iphone ,1 calculator, 1 pen to the bed. 
    observation : I am holding nothing. And I am in office. I see pen and calculator in office. 
    output: go grasp target object <pen>, go grasp target object <calculator>, transport objects I'm holding to the bed.
    Example2: Instruction: Transport 1 iphone ,1 calculator, 1 pen to the bed. 
    observation : I am holding two things on my hand. And I am in office. I see pen and calculator in office. 
    output: transport objects I'm holding to the bed.
    Example3: Instruction: Transport 1 iphone ,1 calculator, 1 pen to the bed. 
    observation : I am holding two things on my hand. And I am in office. I see nothing in office. 
    output: None
    """    

    PIPELINE_STATE["create_action_chain_rules"] =  f"""Based on the information, \
    Please make decision about wheather grasp the target object that in your current room or transport it to the bed.\
    Each robot can hold two things at a time, and they can be objects or containers. Each robot can grasp containers and put objects into them to hold more objects at a time. \
    Note that a container can contain three objects, and will be lost once transported to the bed. Each robot can only put objects into the container it hold after grasping it. \
    1. The first step should be suitable for the actions can be performed. \
    2. {thinking_cot} \
    3. If you don't know where is the bed, just try to go to bedroom. The 'transport' can only be executed when found the bed. \
    {gpt_4o_mini_rule1}
    \n{action_chain_output_requirement}"""

    ###################################################################
    ############### refine method choice ##############################
    ###################################################################


    PIPELINE_STATE["refining_role"] = "You function as an optimal planner in a robotic task planning mission. Please help me improve our current agent's efficiency. "
    PIPELINE_STATE["refining_examples"] = ""
    PIPELINE_STATE["refining_rules"] =   f"""
    **We expect you:**  
    - Fix minor issues (e.g., typos, alignment) while keeping the original intent.  
    - Ensure:
        - All rooms/objects use explicit names + IDs.  
        - Actions follow minimal syntax. 
        {gpt_4o_mini_rule1}
        - {thinking_cot}
    {refine_result_output_requirement}
    """ 


    ###################################################################
    ############### refine action step ################################
    ###################################################################
 
    PIPELINE_STATE["refine_action_step_role"] = "You are an expert in fixing one step of planning."
 
    PIPELINE_STATE["refine_action_step_examples"] = "" 
    _ = """This example from another scene illustrates an effective pattern that can be referenced for the task allocation steps, but the target object may not consist with ours target: 
    instruction: Transport 1 iphone ,1 calculator, 1 pen to the bed. 
    observation: I have explored all of see <Office> , I found pen and calculator in there. 
    Agent 0 previous plan: go to <Office> , explore current room <Office>, go to <Kitchen>,go grasp target object <calculator> ,transport objects I'm holding to the bed. 
    You have failed in step go to <Kitchen>.
    output : go grasp target object <pen>  """    

    PIPELINE_STATE["refine_action_step_rules"] =\
    f"Based on the information, \
    Please make modifications to the step you are failled in while maintaining its overall intention.\
    The new step you create will replace the failed step, so: \
    1. The new step should be suitable for the actions can be performed.\n \
    2. {thinking_cot}\n\
    3. If you don't know where is the bed, just try to go to bedroom. The 'transport' can only be executed when found the bed.\n \
    You should specifically point out the names of the rooms to visit and their IDs, to avoid confusing rooms with the same name.\n \
    Each step only can represent a minimal action, which follow a specific format, such as: go to <room_name>; put <object_name> into the container <container_name>; go grasp target object <object_name>; explore current room <room_name>; transport objects I'm holding to the bed.\n{gpt_4o_mini_rule1}\n{action_output_requirement}"
 
    ###################################################################
    ###################### refine action chain ########################
    ###################################################################
 
    PIPELINE_STATE['action_chain_mutation_role'] = "Review the current action chain and, if a step is infeasible or inefficient, adjust it using a valid available action. Keep the reasoning brief and internal."
    # """Review the current action chain and adjust it to fit available actions while maintaining the overall task objective. For each action, briefly consider its cost and potential benefits internally. Then, if needed, modify or replace the step to improve task efficiency. """

    PIPELINE_STATE['action_chain_mutation_examples'] =  "" """Example:\n
        If a step such as [go grasp target object <object> (id)] proves ineffective, replace it with another valid action while preserving the intended plan. """

    PIPELINE_STATE['action_chain_mutation_rules'] = f"When a step fails, create a replacement step that: \n 1. Clearly identifies any room, container, or object with its name and ID.\n2. Follows the minimal action format (e.g., 'go to explore <room> (id)').\n3. {thinking_cot}.\n{gpt_4o_mini_rule1}\n{action_chain_output_requirement}"




    # =============================================================================
    # Solve conflicting action chains (Between different agents)
    # =============================================================================
    PIPELINE_STATE["solve_conflict_prompt"] = "Agent 0 and Agent 1 have conflicting action chains, meaning their historical and future actions overlap, resulting in unnecessary repetitions, such as repeatedly checking the same container. However, some actions are worth repeating, such as going to the same room together. Therefore, it is necessary to evaluate whether and how to modify their action chains to avoid redundant actions that lack significant value. Each agent can hold up to two objects in hand. \n"
        
    PIPELINE_STATE["solve_conflict_rules"] = f"{thinking_cot}\n{gpt_4o_mini_rule1}\n{conflict_solution_output_requirement}"
    _ = """1. Overlapping but Partially Valuable Steps Situation: Robots A and B need to open the same container repeatedly.\nRobot A: Skip Action Step - Bypass the container-opening if it’s already open and proceed to verification.\nRobot B: Refine Action Step - Adjust the opening with a sensor check or delay.\n2. Redundant Action Sequences in Time-Critical Tasks Situation: Both robots plan to retrieve an item, creating redundancy.\nRobot A: Refine Action Chain - Shorten the sequence by merging steps to save time.\nRobot B: Skip Action Step - Omit unnecessary retrieval steps if the item is en route.\n3. Minor Conflict, One Action Needs AlterationSituation: Both robots move to a destination but have conflicting secondary actions.\nRobot A: None - Keep its action as it adds value.\nRobot B: Refine Action Step - Shift focus to supporting tasks instead of grasp a duplicate item.\n4. Conflict in Time-Sensitive Cooperative Tasks Situation: Both robots must reach a new room quickly, risking delays.\nRobot A: Refine Action Chain \n- Optimize routes to reduce delays.\nRobot B: Skip Action Step \n- Remove non-critical steps for faster movement. General Guidelines for Refinement Refine Action Step: For specific conflicts; alters execution style/timing.\nSkip Action Step: For redundant actions; ensures data retention. Refine Action Chain: For lengthy sequences; improves time efficiency. These strategies enhance robot operations, ensuring efficient cooperation while minimizing conflicts and redundancy."""
    # The updated PIPELINE_STATE object is now complete.
    PIPELINE_STATE['task_allocation_result'] = dict()
    for robot_id in range(ROBOT_NUMBER):
        PIPELINE_STATE['task_allocation_result'][robot_id] = ""


    return PIPELINE_STATE


def get_scene_overall_description(batch_prompt_elements :BATCH_PROMPT_Constituent_Elements):
    """
    Converts the game state data into a natural language description in English.

    Args:
    prompt_elements: A dictionary containing the game state information.

    Returns:
    A string representing the natural language description of the game state.
    """
    description = ""
    batches = batch_prompt_elements.batch_prompt_constituent_elements
    for  data in batches:

        description += f"Agent {data.agent_name} (id:{get_name_to_id(data.agent_name)}) is in {data.current_room} now. {construct_holding_description(data.grabbed_objects)}\n {convert_discovery_to_text(data.discovery)}" 
        # description += f"Self-description of Agent {data.agent_name} (id:{get_name_to_id(data.agent_name)} ):\n{data.progress_desc}\n"
    return description

def get_agents_description(batch_prompt_elements :BATCH_PROMPT_Constituent_Elements):
    description = []
    batches = batch_prompt_elements.batch_prompt_constituent_elements
    for data in batches:
        description.append(data.progress_desc)
        description.append(f"It's action space: {data.available_plans}")
    return "\n".join(description)

def get_agent_obs(batch_prompt_elements :BATCH_PROMPT_Constituent_Elements):
    description = []
    batches = batch_prompt_elements.batch_prompt_constituent_elements
    for  data in batches:
        description.append(f"Self-description of agent {data.agent_name} (id:{get_name_to_id(data.agent_name)} ):\n{data.progress_desc}\n")
    return "".join(description)
