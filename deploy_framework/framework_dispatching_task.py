# from sympy import N
from framework_common_imports import *
from config import display_inference_process



def wrap_task_assignment_result(text: str):
    result = None
    json_data_list = extract_last_json(text)
    for json_data in json_data_list:
        if json_data is not None and "robot_id_task_pairs" in json_data and "reason" in json_data and "action_chain" in json_data["robot_id_task_pairs"][0]:
            # 创建 Task_Decomposing_Result 的实例
            robot_tasks = [
                Task_Decomposing_Result(
                    robot_id=pair["robot_id"],
                    task_content=Action_Chain(intention="",chain=[Action(step = step, reason = "") for step in pair["action_chain"]])
                ) for pair in json_data["robot_id_task_pairs"]
            ]

            # 创建 Task_Decomposing_Struct 的实例
            result = Task_Decomposing_Struct(
                robot_id_task_pairs=robot_tasks,
                reason=json_data["reason"]
            )

            # 输出结果
            log_system.PRINT("wrap_task_assignment_result:\n",result)
            break
    else:
        print("未找到有效的 JSON 数据。")
    return result

from interface_wrapper import get_task_assignment_output_rules

# 组装 prompt

def get_decomposed_dispatching_prompt(state: AGENT_STATE,
                                      to_be_allocated_agents_id=[], executeable_agent_info : list = []):
    prompts = [
        state["task_assignment_role"],"\nHere is the infomations:\n",
        # state["task_assignment_example"],
        "\n".join(state['overall_observation']),
        f"Our goal is : {state['human_instruction']}",
    ]

    if len(to_be_allocated_agents_id) == 2:
        temp = " and ".join([f'Agent {id}' for id in to_be_allocated_agents_id])
        prompts.append(f"Please allocate tasks to agent : {temp}. ")
        prompts.append(get_task_assignment_output_rules("\n".join(state['overall_observation'])))
    else:
        agent_id = to_be_allocated_agents_id[0]
        i_agent_id, i_action_chain = executeable_agent_info[0]
        temp = f"You need to allocate task to agent {agent_id}. And the allocation action chain should be careful to avoid the repeat action chain {i_action_chain} from agent {i_agent_id}. "
        prompts.append(temp)
        prompts.append(get_task_assignment_output_rules(state['overall_observation'][agent_id]))
    return "".join(prompts)

# 调用大模型




def new_dispatching_task(state: AGENT_STATE,agents_:list[Agent]=[]) -> Task_Decomposing_Struct:

    to_be_allocated_agents_id = [agent.agent_id for agent in agents_ if agent.stage == Stage.dispatching_task or agent.stage == Stage.initial_dispatching_task]

    executeable_agent_info = []
    for agent in agents_:
        if agent.stage != Stage.dispatching_task and agent.stage != Stage.initial_dispatching_task:
            history_nodes, current_block_nodes, next_block_nodes = agent.action_tree.get_history_next_two_blocks_action_chain(change_times=2)
            nodes = []
            nodes.extend(history_nodes)
            nodes.extend(current_block_nodes)
            nodes.extend(next_block_nodes)
            executeable_agent_info.append((agent.agent_id," -> ".join(nodes)))


    prompt = get_decomposed_dispatching_prompt(state, to_be_allocated_agents_id , executeable_agent_info)

    log_system.PRINT("------------------ dispatching_task ------------------")
    log_system.PRINT(prompt)
    ans = asyncio.run(wrap_invoke(prompt,"dispatching_task",wrap_task_assignment_result))
    log_system.PRINT(ans)


    log_system.LOG("------------------ dispatching_task Q------------------")
    log_system.LOG(prompt)
    log_system.LOG("------------------ dispatching_task A------------------")
    log_system.LOG(ans)



    return ans