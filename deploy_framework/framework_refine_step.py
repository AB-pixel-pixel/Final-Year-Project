from framework_common_imports import *
from typing import Any


def wrap_action(text: str) -> Optional[Action]:
    """Extract action and reason from the output string using regex."""
    result = None
    
    json_data_list = extract_last_json(text)
    for json_data in json_data_list:
        if json_data is not None and "step" in json_data and "reason" in json_data:
            result =Action(step=json_data["step"], reason=json_data["reason"])
            # 输出结果
            print(result)
            break
        else:
            log_system.PRINT("未找到有效的 JSON 数据. ")
        
    return result
   
def filter_same_action(text:str , action_chain :list[Action]):
    if len(action_chain) != 0:
        lastest_action = action_chain[-1].step
        if lastest_action != text:
            action_chain.append(Action(step = text, reason = ""))
    else:
        action_chain.append(Action(step = text, reason = ""))

    return action_chain




def wrap_action_chain(text: str) -> Optional[Action_Chain]:
    result = None

    json_data_list = extract_last_json(text)
    log_system.PL("json_data from wrap_action_chain:\n",json_data_list)
    for json_data in json_data_list:
        if json_data is not None and "action_chain" in json_data:
            action_chain = []
            for step in json_data["action_chain"]:
                if ';' in step:
                    steps = step.split(';')
                    for step in steps:
                        action_chain = filter_same_action(step, action_chain)
                elif ',' in step:
                    steps = step.split(',')
                    for step in steps:
                        action_chain = filter_same_action(step, action_chain)
                else:
                    action_chain = filter_same_action(step, action_chain)
            result = Action_Chain(intention=json_data.get("intention"),chain=action_chain)
            break
        else:
            log_system.PRINT("未找到有效的 JSON 数据. ")
    return result

def wrap_refine_result(text: str):
    result = None

    json_data_list = extract_last_json(text)
    log_system.PL("json_data from wrap_refine_result:\n",json_data_list)

    for json_data in json_data_list:
        if json_data is not None and "action_chain" in json_data and "inference_process" in json_data:
            action_chain=Action_Chain(intention="",chain=[Action(step = step, reason = "") for step in json_data["action_chain"]])
            result = refineResult(inference_process=json_data['inference_process'], action_chain=action_chain)
            log_system.PL("result: ", result)
            break
    return result

def wrap_conflict_solution(text : str):
    result = None

    json_data_list = extract_last_json(text)
    for data in json_data_list:
        if data is not None and "action_chain0" in data and "action_chain1" in data:
            action_chain0 = Action_Chain(chain = [Action(step = step, reason = "") for step in data["action_chain0"]],intention = data.get("reason_agent0"))
            action_chain1 = Action_Chain(chain = [Action(step = step, reason = "") for step in data["action_chain1"]],intention = data.get("reason_agent0"))
            
        
            result = ConflictSolution(
                method_agent0=data.get("method_agent0"),
                reason_agent0=data.get("reason_agent0"),
                action_chain0 = action_chain0,
                method_agent1=data.get("method_agent1"),
                reason_agent1=data.get("reason_agent1"),
                action_chain1=action_chain1
            )
            break

    log_system.PRINT("result: ", result)
    return result

from interface_wrapper import get_refine_action_step_rules

def get_refine_step_prompt(state : AGENT_STATE, task_description, action_chain, available_plans :str,planning_reason):
    available_plans = available_plans.replace('\n','    ')
    prompt = [state['refine_action_step_role'], f"Our goal is {state['current_task']}. ",
              state["refine_action_step_examples"], 
    # f"Here is observation from robot: {state['observation']} ", 
    action_chain,
    planning_reason,
    f"It's action space the following actions: {available_plans}. ", 
    get_refine_action_step_rules(state['observation'])]
    return "".join(prompt)

def action_step_mutation(agent : Agent) -> Tuple[str,Any]:
    """ 因为动作步骤不在可选动作之中，修改单一的步骤 """
    log_system.PRINT(""" 因为动作步骤不在可选动作之中，修改单一的步骤 """)
    state :AGENT_STATE = agent.state
    task_content = state['current_task']
    available_plans = state['available_plans']
    planning_reason = agent.get_disposable_planning_reason()

    action_chain = agent.action_tree.get_refine_action_step_info()

    prompt = get_refine_step_prompt(state,task_description=task_content,action_chain=action_chain,available_plans= available_plans,planning_reason = planning_reason)
    
    log_system.PL(prompt)
    ans = asyncio.run(wrap_invoke(prompt,"refine_step",wrap_action))
    log_system.PL(ans)

    return prompt,ans



def convert_action_chain_to_text(action_chain : Action_Chain) -> str:
    plan =  " -> ".join([action.step for action in action_chain.chain])
    ans = f"{plan}. The intention of this action chain is:{action_chain.intention}"
    return ans

from interface_wrapper import get_create_action_chain_rules

def create_action_chain(agent : Agent, available_plans:str) -> Tuple[str,Any]:
    """面对未知环境中的发现进行新的规划"""
    log_system.PRINT("------------------ 面对未知环境中的发现进行新的规划 ------------------")

    state :AGENT_STATE = agent.state
    task_content = state['current_task']
    planning_reason = agent.get_disposable_planning_reason()

    action_chain = agent.action_tree.get_create_action_chain_info(10)

    create_action_chain_rules = get_create_action_chain_rules(state['observation'])
    prompt = [
    state["create_action_chain_role"], f"Our goal is {state['current_task']} ",
    # f"My plan is: {state['task_content']}. ",
    f"Provided information : {state['observation']}. ".replace("\n.","\n"),
    # f"My action space: {available_plans}. ",
    f"{action_chain}. "
    f"{planning_reason}. ",
    state["create_action_chain_examples"],
    create_action_chain_rules
    ]
    prompt = "".join(prompt)
    prompt = prompt.replace("..",".")
        
    log_system.PL('\n',prompt)
    ans = asyncio.run(wrap_invoke(prompt,"create_action_chain",wrap_action_chain))
    log_system.PL('\n',ans)



    return prompt, ans

from interface_wrapper import get_action_chain_mutation_rules

def assemble_action_chain_mutation_prompt(state : AGENT_STATE,task_description,available_plans,planning_advice,negative, action_chain_info):
    prompt = [state['action_chain_mutation_role'],
    f"Our goal is : {state['current_task']}.",  
    f"It's observation is : {state['observation']}. ", 
    f"{action_chain_info}",
    f"It's action space the following actions: {available_plans}. ", 
    planning_advice if planning_advice.endswith(".") else planning_advice+". ",
    state["action_chain_mutation_examples"], 
    get_action_chain_mutation_rules(state['observation'])]
    prompt = "".join(prompt)
    return prompt


def action_chain_mutation(agent: Agent,available_plans:str, negative : Optional[List[Tuple[str, str]]],  action_chain_info : str ):
    log_system.PRINT("------------------ 修改整个动作链条 ------------------")

    state :AGENT_STATE = agent.state
    task_content = state['current_task']
    advice = agent.get_disposable_planning_reason()

    prompt = assemble_action_chain_mutation_prompt(state,task_description=task_content,available_plans = available_plans, planning_advice = advice, negative = negative, action_chain_info = action_chain_info)

    log_system.PL("------------------ ACTION CHAIN mutation ------------------")
    log_system.PL(str(prompt))

    ans = asyncio.run(wrap_invoke(prompt,"action_chain_mutation", wrap_action_chain))

    log_system.PL(str(ans))



    # 保存相关信息
    
    return prompt, ans

from interface_wrapper import get_refining_rules

def refining(agent: Agent, action_chain_info: str = "") -> Tuple[str,Any]:
    """ using different to refine action chain in various sizes  """
    log_system.PRINT("------------------ 进行修改方案的选择 ------------------")

    state : AGENT_STATE = agent.state
    task_content = state['current_task']
    advice = agent.get_disposable_planning_reason()
    advice = f"The problem is : {advice}.".replace("..",".")
    available_plans = agent.state['available_plans_with_info']
    action_chain_info = agent.action_tree.get_refine_action_info()

    refining_rules =  get_refining_rules(state['observation'])

    prompt = state['refining_role'] +   f"Ours goal is : {task_content}. " + state['observation']  + action_chain_info + advice + refining_rules
    log_system.PL("------------------ REFINE METHOD CHOICE ------------------")
    log_system.PL(prompt)
    ans = asyncio.run(wrap_invoke(prompt,"refining",wrap_refine_result)) # refineResult
    log_system.PL(str(ans))

    # 保存相关信息
    
    return prompt, ans 

def matching_refine_method_v2(choice : str) -> str:
    method = choice.lower()
    ans = ""
    if 'none' in method:
        ans = 'none'
    elif 'chain' in method or 'skip' in method:
        ans = 'refine_action_chain'
    elif 'step' in method:
        ans = 'refine_action_step'
    else:
        ans = 'refine_action_chain'
        # raise ValueError(f"Unrecognized refine method: {method}")
    log_system.PL(f"matching_refine_method: \ninput: {method}\n matching: {ans} \n")
    return ans

from interface_wrapper import get_solve_conflict_rules

def solve_confilct(agent: Agent, conflict_agent : Agent, conflict_message : str) -> Tuple[str, Any]:
    """ 提出冲突方案 """
    log_system.PRINT("------------------ 提出冲突方案 ------------------")
    state : AGENT_STATE = agent.state
    task_content = state['current_task']
    advice = agent.get_disposable_planning_reason()
    available_plans = agent.state['available_plans']
    action_chain_info = agent.action_tree.get_refine_action_info()

    prompt = state['solve_conflict_prompt'] + f"Common goals are : {task_content}" 
    prompt += f"Agent {agent.agent_id}: " + state['observation']  + action_chain_info

    prompt += f"Agent {conflict_agent.agent_id}: " + conflict_agent.state['observation'] + f"It's action space actions: {conflict_agent.state['available_plans']} "
    prompt += f"Conflict message: {conflict_message} "

    conflict_action_info = conflict_agent.action_tree.get_refine_action_info()
    prompt += conflict_action_info
    prompt += get_solve_conflict_rules(state['observation'])

    log_system.PL("------------------ conflict solution ------------------")
    log_system.PL(str(prompt))
    ans = asyncio.run(wrap_invoke(prompt,"conlict_solution",wrap_conflict_solution)) # ConflictSolution
    log_system.PL(str(ans))


    # 保存相关信息
    
    return prompt, ans # ConflictSolution


if __name__ == "__main__":
    result = None
    json_data_list = [{'inference_process': 'Since I can hold two objects at a time and I am currently empty-handed, I will focus on grasping objects that are available in the Livingroom (8000), where I found apples. I will prioritize grasping the apples first, as they are part of the target items I need to transport to the bed. After grasping, I will either continue to grasp additional items or, if I reach full capacity, I will prepare to transport them to the bed if it is discovered.', 'intention': "To efficiently gather the target objects (especially the apples first) and transport them to the bed once the bed's location is discovered.", 'action_chain': ['go grasp target object <apple> (3205029)', 'go grasp target object <apple> (177311)']}]
    for json_data in json_data_list:
        if json_data is not None and "action_chain" in json_data:
            action_chain = []
            for step in json_data["action_chain"]:
                if ';' in step:
                    steps = step.split(';')
                    for step in steps:
                        action_chain = filter_same_action(step, action_chain)
                elif ',' in step:
                    steps = step.split(',')
                    for step in steps:
                        action_chain = filter_same_action(step, action_chain)
                else:
                    action_chain = filter_same_action(step, action_chain)
            result = Action_Chain(intention=json_data.get("intention"),chain=action_chain)
            print(result)
            break
        else:
            log_system.PRINT("未找到有效的 JSON 数据. ")
    print(result)