from framework_structure import *
# Result
# 62.7 STEP

task_assignment_output_rules =  """
Please generate the output in the following json format:
{
  "reason": "<specific reason explanation>",
  "robot_id_task_pairs": [
    {
      "robot_id": <robot_id>,
      "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
      ]
    },
    {
      "robot_id": <robot_id>,
      "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
      ]
    },
  ]
}
"""

action_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the action.",
    "reason": "The reason for choosing the action.",
    "step":  "The chosen action."
}
"""

action_chain_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the action chain.",
    "intention": "The intention or reason of the action chain.",
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

refine_result_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the better action chain.",
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

"""
Please generate the output in the following json format:
{
    "reason": "The reason for choosing the method. ",
    "method": "The method of refining. ",
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

conflict_solution_output_requirement = """
Please generate the output in the following json format:
{
    "reason_agent0": "The reason for agent0 choosing the method. ",
    "method_agent0": "The method of agent0 refining. ",
    "action_chain0": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ],
    "reason_agent1" : "The reason for agent1 choosing the method. ",
    "method_agent1": "The method of agent1 refining. ",
    "action_chain1": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ],
}
"""

action_output_requirement = """
Please generate the output in the following json format:
{
    "inference_process": "The inference process of the action.",
    "reason": "The reason for choosing the action.",
    "step":  "The chosen action."
    "action_chain": [
        "[action] <object_name> (object_id)",
        "[action] <object_name> (object_id)"
    ]
}
"""

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
    for  data in batches:
        description.append(f"Agent {data.agent_name} (id:{get_name_to_id(data.agent_name)}) is in {data.current_room} now. {construct_holding_description(data.grabbed_objects)}\n {convert_discovery_to_text(data.discovery)}")
        description.append(f"It's action space: {data.available_plans}")
    return "".join(description)

def get_agent_obs(batch_prompt_elements :BATCH_PROMPT_Constituent_Elements):
    description = []
    batches = batch_prompt_elements.batch_prompt_constituent_elements
    for  data in batches:
        description.append(f"Self-description of agent {data.agent_name} (id:{get_name_to_id(data.agent_name)} ):\n{data.progress_desc}\n")
    return "".join(description)

def get_initial_public_pipeline() -> dict:
    """
    Converts the game state data into a natural language description in English.
    Args:
    prompt_elements: A dictionary containing the game state information.

    Returns:
    A string representing the natural language description of the game state.
    """
    # batch_prompt_elements = batch_prompt_elements.batch_prompt_constituent_elements

    PIPELINE_STATE = dict()

    PIPELINE_STATE['task_allocation_result'] = dict()
    for robot_id in range(ROBOT_NUMBER):
        PIPELINE_STATE['task_allocation_result'][robot_id] = ""

    ###################################################################
    ######################### task assignment #########################
    ###################################################################
 
    PIPELINE_STATE["task_assignment_role"] = "You are provided with a detailed task assignment. For each identified action:\n- Briefly consider its cost and benefit internally as you plan.\n- Assign tasks to agents to distribute initial exploration efficiently.\n- When the agent enters a new room, include the command 'replan' to adjust actions based on current observations.\n- Agents can hold up to two objects; refer to objects explicitly by their name and ID (e.g., <table> (712)). " # Proceed directly with the task assignment.
 
    PIPELINE_STATE["task_assignment_example"] = ""
 
    PIPELINE_STATE['task_assignment_output_rules'] = task_assignment_output_rules

 
 
    ###################################################################
    ###################### Create action chain ########################
    ###################################################################
    
    
    PIPELINE_STATE["create_action_chain_role"] = "Develop a new action chain based on the latest state, incorporating discovered elements. While planning, briefly consider each step's internal cost and benefit, but do not output explicit scores."
    PIPELINE_STATE["create_action_chain_examples"] = """Example:"
    "Instruction: Place found items on <kitchentable> (130) after exploring the kitchen.
    Output:
    {
    "inference_process": "Based on the current observation, the agent needs to gather specific items and place them on the coffee table. The agent will first check the cabinet for any items, then explore the kitchen to find the remaining items before placing them on the coffee table.",
    "intention": "Since I am only hold two objects , I will pick up two and then put them down each times.",
    "action_chain": [
        "[gocheck] <cabinet> (216)", 
        "[gograsp] <apple> (375)", 
        "[goexplore] <kitchen> (11)",
        "[gograsp] <pudding> (376)", 
        "[goput] <coffeetable> (268)", 
        "[gograsp] <juice> (377)", 
        "[gograsp] <cupcake> (378)", 
        "[goput] <coffeetable> (268)", 
        "replan"
    ]
    }"""
    PIPELINE_STATE["create_action_chain_rules"] = f"\n1. Honors the two-object limit per agent.\n2. Provides room names and IDs to eliminate ambiguity.\n3. Uses the correct action formats.\n4. Explain the reasoning for each step.\n5. Is as concise as possible while achieving the task.\n{action_chain_output_requirement}"
 
 
        
    ###################################################################
    ############### refine method choice ##############################
    ###################################################################
 
 
    PIPELINE_STATE["refining_role"] = "You function as an optimal planner in a robotic task planning mission. Please provide a plan with efficient. "
    PIPELINE_STATE["refining_examples"] = ""
    PIPELINE_STATE["refining_rules"] = f"""
    **Simplify Actions: Optimize action sequences for efficiency:**

    1. **Historical Action Information and Future Action Plans**
    - I will provide some historical action information and future action plans. Your task is to modify the future actions based on this information.
    - You can choose to retain some excellent action sequences by only modifying the next step, or you can make significant changes.

    2. **Modification Scope**
    - **Minor Modifications**: Adjust only the current action, ensuring the overall task goal remains unchanged.
    - **Major Modifications**: Reconstruct the action sequence, redesign the next step, or even replace the entire action chain, ensuring execution effectiveness and feasibility.

    3. **Optimization Advice**
    - Fix minor issues (e.g., typos, alignment).
    - Briefly evaluate alternatives for efficiency (consider: The agent holds ≤2 objects at once).
    - Shorten or adjust the sequence if it’s inefficient or broken.
    - Explain your choice in 1-2 sentences.
    - Ensure:
        - All rooms/objects use explicit names + IDs.
        - Task goals remain intact.
    {refine_result_output_requirement}
    """
 
    ###################################################################
    ############### refine action step ################################
    ###################################################################
 
    PIPELINE_STATE["refine_action_step_role"] = "Coordinate with Alice for a joint task. For a given action step, consider alternative actions for better efficiency and resource management (considering that an agent can hold only two objects). Choose the best option based on internal evaluation, and after adjustment, clearly state the chosen action following the minimal action syntax. "
 
    PIPELINE_STATE["refine_action_step_examples"] = ""
    PIPELINE_STATE["refine_action_step_rules"] = f" Consider holding capacity and actionable efficiency. Select the action by its description and provide your reasoning step-by-step. {action_output_requirement}"
 
 
    ###################################################################
    ###################### refine action chain ########################
    ###################################################################
 
    PIPELINE_STATE['action_chain_mutation_role'] = "Review the current action chain and, if a step is infeasible or inefficient, adjust it using a valid available action. Keep the reasoning brief and internal. Ensure that:1. The agent's carrying capacity (up to two objects) is not exceeded.2. All objects, rooms, or containers are explicitly identified by name and ID.3. The action format stays minimal (e.g., [gocheck] <container (id)>).4. The revised chain maintains overall task objectives." 
        
    PIPELINE_STATE['action_chain_mutation_examples'] = " Example: If [gocheck] <container> (id) is not feasible, consider a replacement action that still inspects a relevant container."
    
    PIPELINE_STATE['action_chain_mutation_rules'] = \
    f"When a step fails, create a replacement step: \n 1. Uses one of the allowed actions.\n2. Considers that an agent can only hold two objects at a time.\n3. Identifies any room, container, or object with its name and ID.\n4. Follows the minimal action format (e.g., '[goexplore] <room> (id)').\n5. Provide your reasoning from the perspective of completing the task efficiently.\n6. Ideally, use a shorter, more direct chain to achieve the goal. {action_chain_output_requirement}"
    
    
 
 
    ###################################################################
    ######################### Solve conflict #########################
    ###################################################################
 
 
    PIPELINE_STATE["solve_conflict_prompt"] = "Agent 0 and Agent 1 have conflicting action chains, meaning their historical and future actions overlap, resulting in unnecessary repetitions, such as repeatedly checking the same container. However, some actions are worth repeating, such as going to the same room together. Therefore, it is necessary to evaluate how to modify their action chains to avoid redundant actions that lack significant value and to prevent the same actions from being executed in the short term. Each agent can hold up to two objects in hand. Please evaluate the cost of each action in the thought process before moving on to planning.\n "
    
    PIPELINE_STATE["solve_conflict_rules"] = f"1. Overlapping but Partially Valuable Steps Situation: Robots A and B need to open the same container repeatedly.Robot A: Skip Action Step - Bypass the container opening if it’s already open and proceed to verification.Robot B: Refine Action Step - Adjust the opening with a sensor check or delay.2. Redundant Action Sequences in Time-Critical TasksSituation: Both robots plan to retrieve an item, creating redundancy.Robot A: Refine Action Chain - Shorten the sequence by merging steps to save time.Robot B: Skip Action Step - Omit unnecessary retrieval steps if the item is en route. 3. Minor Conflict, One Action Needs AlterationSituation: Both robots move to a destination but have conflicting secondary actions. Robot A: None - Keep its action as it adds value. Robot B: Refine Action Step - Shift focus to supporting tasks instead of picking up a duplicate item.4. Conflict in Time-Sensitive Cooperative TasksSituation: Both robots must reach a new room quickly, risking delays. Robot A: Refine Action Chain - Optimize routes to reduce delays.Robot B: Skip Action Step - Remove non-critical steps for faster movement. General Guidelines for RefinementRefine Action Step: For specific conflicts, alter execution style/timing. Skip Action Step: For redundant actions, ensure data retention. Refine Action Chain: For lengthy sequences, it improves time efficiency. None: When actions support coordinated behavior without inefficiency. These strategies enhance robot operations, ensuring efficient cooperation while minimizing conflicts and redundancy. {conflict_solution_output_requirement}"
 
    return PIPELINE_STATE

if __name__ == "__main__":
    print(get_initial_public_pipeline())