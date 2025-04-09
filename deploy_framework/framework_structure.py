
from shapely import total_bounds
from sklearn.metrics import rand_score
from process_text import *

import requests
import inspect
import json
import os
import operator
import time
import csv
from pprint import pprint
import re
import copy
from collections import deque
from enum import Enum
from typing import Annotated,Union,Optional, Tuple ,List,Dict
import unittest
import asyncio
from click import File
from pydantic import BaseModel, Field
from regex import D
from typing_extensions import TypedDict
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import textwrap
from pathlib import Path


from langchain_community.callbacks.manager import get_openai_callback

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import uuid

import pandas as pd


from communication_protocol import *


from LOG_SYSTEM import LOG_SYSTEM, log_system






def is_location(text : str) -> bool:
    return "room" in text or "kitchen" in text or "office" in text
# Input
ROBOT_NUMBER = 2

# TASK




# Action Chain  
class Action(BaseModel):
    step: str = Field(description="The action")
    reason: str = Field("Thinking process of choosing the correct action.",description="The thinking process of choosing the correct action.")


class Action_Chain(BaseModel):
    chain: List[Action] = Field(description="A sequence of action.")
    intention : str = Field(description="The intention or reason of the action chain.")

class Task_Decomposing_Result(BaseModel):
    robot_id : int = Field(description="The id of the robot.")
    # reason : str
    task_content : Action_Chain = Field(description="The task content.")



class Task_Decomposing_Struct(BaseModel):
    robot_id_task_pairs : list[Task_Decomposing_Result]  = Field(description="Task breakdown for each robot.")
    reason : str = Field(description="The reason for task breakdown.")


def to_description(task_decomposing_struct : Task_Decomposing_Struct):
    description = {
        "robot_tasks": [],
        # "unassigned_tasks": task_decomposing_struct.the_rest_tasks,
        # "all_tasks_assigned": task_decomposing_struct.all_tasks_assigned
    }
    for pair in task_decomposing_struct.robot_id_task_pairs:
        description["robot_tasks"].append({
            "robot_id": pair.robot_id,
            "task_content": pair.task_content
        })
    return json.dumps(description)

def action_chain_to_text(action_chain : Action_Chain):
    return " -> ".join([f"{action.step}" for action in action_chain.chain])


class Action_Option(BaseModel):
    action: str = Field("The option of the availble actions")


class refineMethodChoice(BaseModel):
    method: str = Field("The method of refining")
    reason: str = Field("The reason for choosing the method")

class refineResult(BaseModel):
    # method: str = Field("The method of refining")
    inference_process: str = Field("The reason for choosing the method")
    action_chain : Action_Chain = Field("The refined action chain")
 
class ConflictSolution(BaseModel):
    method_agent0: str = Field("The method for agent 0 to refine the action chain")
    reason_agent0: str = Field("The reason for agent 0 choosing this method")
    action_chain0: Action_Chain = Field("The advice for agent 0 to adjust the action chain, if not nessary, leave it None")
    method_agent1: str = Field("The method for agent 1 to refine the action chain")
    reason_agent1: str = Field("The reason for agent 1 choosing this method")
    action_chain1: Action_Chain = Field("The advice for agent 1 to adjust the action chain, if not nessary, leave it None")

class ConflictSolution_V3(BaseModel):
    inference_process : str = Field("The inference process of the conflict solution")
    solution : str = Field("The solution of the conflict")
    reason : str = Field("The reason for the solution")
    Agent0 : Action_Chain = Field("The action chain of agent 0")
    Agent1 : Action_Chain = Field("The action chain of agent 1")


# aiding function 

def overwrite_plans(old_tasks : Action_Chain,new_tasks : Action_Chain, index: int ):
    new_chain = new_tasks.chain
    length = len(new_chain)
    ans_tasks = copy.deepcopy(old_tasks)
    for i in range(length):
        if i + index < len(ans_tasks.chain):        
            ans_tasks.chain[i+index] = new_chain[i]
        else:
            ans_tasks.chain.append(new_chain[i])
    return ans_tasks


import tiktoken
from config import model_name
if 'gpt' in model_name:
    TOKEN_ENCODING = tiktoken.encoding_for_model(model_name)
    def count_tokens(text) -> int:
        if not isinstance(text,str):
            text = text.json()
        return len(TOKEN_ENCODING.encode(text))




# ------------------------------- 大模型输出结构设置 --------------------------



from config import large_model_server_ip_port, model_name

if 'gpt' in model_name:
    from config import api_key4_0_key, model_name, k_temperature, k_top_p
    from openai import OpenAI
    GLOBAL_LLM = OpenAI(
        base_url='https://xiaoai.plus/v1',
        # sk-xxx替换为自己的key
        api_key=api_key4_0_key
    )
else:
    from communication_protocol import AIRS_LLM
    GLOBAL_LLM = AIRS_LLM(large_model_server_ip_port)


# ------------------------------- 大模型输出结构设置 End --------------------------



# Overall state
class AGENT_STATE(TypedDict):
    # 共用
    human_instruction : str
    adaptive_prompt : Annotated[list, operator.add]
    static_system_prompt : dict[str,str]
    overall_observation : str  # 需要公共修改的屬性，用於task planning
    examples : list
    task_allocation_result: dict[int,str] # 存儲誰正在做什麼的內容，可以加入到prompt之中
    current_task : str     # 當前正在執行的task
    step_num : int
    task_allocation_counter : int

    # private
    # CoT
    id : int 
    
    observation : str  # 觀測狀態
    state : str    # 當前用於存放目標
    msg_from_others : str
    
    frozen_state : str

    chain_of_tasks : Optional[Action_Chain]
    cot_counter : int
    current_cot_step : Union[Action, None]
    need_replan : bool

    task_content : str
    step_content : str
    available_plans : str

    replan_reason : str
    action_history : List[str]
    
    # Prompt
    task_assignment_role: str
    task_assignment_example: str
    task_assignment_output_rules : str
    generate_action_sequence_role: str
    generate_action_sequence_examples: str
    generate_action_sequence_rules: str
    action_chain_mutation_role: str
    action_chain_mutation_examples: str
    action_chain_mutation_rules: str
    create_action_chain_role: str
    create_action_chain_examples: str
    create_action_chain_rules: str
    refine_action_step_role: str
    refine_action_step_examples: str
    refine_action_step_rules: str
    refine_method_choice_role: str
    refine_method_choice_examples: str
    refine_method_choice_rules: str
    refining_role : str
    refining_examples : str
    refining_rules : str
    solve_conflict_prompt : str
    solve_conflict_rules : str
    available_plans_with_info : str
    task_relevant_obsevation : str




def get_initial_agent_state() -> AGENT_STATE:
    """ 返回一个初始的AGENT_STATE对象 """
    return AGENT_STATE(
        human_instruction="",  # 人类指令
        adaptive_prompt=[],  # 适应性提示
        static_system_prompt={},  # 静态系统提示
        overall_observation="",  # 总体观察
        examples=[],  # 示例
        task_allocation_result={},  # 任务分配结果  # 当前任务
        current_task="",
        step_num=0,  # 步骤编号
        task_allocation_counter=0,  # 任务分配计数器
        id=0,  # ID
        observation="",  # 观察
        state="",  # 状态
        msg_from_others="",  # 来自他人的消息
        frozen_state="",  # 冻结状态
        chain_of_tasks=None,  # 根据具体实现提供初始值
        cot_counter=0,  # cot计数器
        current_cot_step=None,  # 当前cot步骤
        task_content="",  # 任务内容
        step_content="",  # 步骤内容
        available_plans="",  # 可用计划
        replan_reason="",  # 重新规划原因
        action_history=[],  # 动作历史
        task_assignment_role="",  # 任务分配角色
        task_assignment_example="",  # 任务分配示例
        task_assignment_output_rules="",  # 任务分配输出规则
        generate_action_sequence_role="",  # 生成动作序列角色
        generate_action_sequence_examples="",  # 生成动作序列示例
        generate_action_sequence_rules="",  # 生成动作序列规则
        action_chain_mutation_role="",  # 动作链变异角色
        action_chain_mutation_examples="",  # 动作链变异示例
        action_chain_mutation_rules="",  # 动作链变异规则
        create_action_chain_role="",  # 创建动作链角色
        create_action_chain_examples="",  # 创建动作链示例
        create_action_chain_rules="",  # 创建动作链规则
        refine_action_step_role="",  # 精炼动作步骤角色
        refine_action_step_examples="",  # 精炼动作步骤示例
        refine_action_step_rules="",  # 精炼动作步骤规则
        refine_method_choice_role="",  # 精炼方法选择角色
        refine_method_choice_examples="",  # 精炼方法选择示例
        refine_method_choice_rules="",  # 精炼方法选择规则
        refining_role="",  # 精炼角色
        refining_examples="",  # 精炼示例
        refining_rules="",  # 精炼规则
        solve_conflict_prompt="",  # 解决冲突提示
        solve_conflict_rules="",  # 解决冲突规则
        available_plans_with_info = "",
        task_relevant_obsevation = "",
        need_replan = False
    )
# ------------------------------智能體數據結構 START----------------------------

class Stage(Enum):
    initial_dispatching_task = 1
    dispatching_task = 2
    initial_generate_action_sequence = 3
    generate_action_sequence = 4  # 生成動作序列
    get_action = 5
    check_action = 6  # 初始化需要執行的動作
    execute_action = 7 # 正在執行動作，需要避免最終的動作生成
    refine_one_step = 8
    refine_steps = 9
    wait = 10





# -------------------------------- Action Tree Start -------------------------------------------


# 定义一个枚举类，表示动作节点的状态
class ActionNodeState(Enum):
    # 数字大小表示覆盖顺序（优先级）
    head_node = 0
    # 动作节点未执行
    not_executed = 1
    # 动作节点无效
    invalid = 2
    # 动作结点已经执行过（ 给 replan 专用 )
    executed = 3
    # 动作节点有效
    valid = 4



def node_state_max(state0: Optional[Union[ActionNodeState, int]], state1: Optional[Union[ActionNodeState, int]]) -> Union[int,ActionNodeState]:
    # 提取状态值，考虑可能为 None 的情况
    value0 = state0.value if isinstance(state0, ActionNodeState) else state0
    value1 = state1.value if isinstance(state1, ActionNodeState) else state1
    # 处理 None 值
    if value0 is None:
        return ActionNodeState(value1)
    if value1 is None:
        return ActionNodeState(value0)

    # 使用枚举值进行比较
    _next_state_value = max(value0, value1)

    return _next_state_value


class ActionNodeType(Enum):
    not_mask = 0
    mask = 1
    unknown = 2
    head_node = 3


class ActionNode:
    def __init__(self, node_id: int, block_id: int, node_type: ActionNodeType, state: ActionNodeState,
                 content: str, effect: str, intention: str, children: Optional[List['ActionNode']] = None,
                 parent_node_id: Optional['ActionNode'] = None, depth :int = 0):
        self.node_id : int = node_id  # Unique identifier for the node
        self.block_id : int = block_id  # Block identifier to categorize nodes
        self.node_type : ActionNodeType = node_type  # Type of the node (e.g., action, task)
        self.state : ActionNodeState = state  # Current state of the node (e.g., not_executed, valid, invalid)
        self.content : str = content  # Content or description of the node
        self.effect : str = effect  # Effect or reason associated with the node
        self.intention : str = intention  # Intention of the block
        self.children : List['ActionNode'] = children if children is not None else []  # List of child nodes
        self.parent_node_id : int  = parent_node_id  # Parent node ID
        self.depth : int  = 0

    def add_child(self, child: 'ActionNode'):
        """Add a child ActionNode to the current node."""
        self.children.append(child)
        child.parent_node_id = self.node_id
        child.depth = self.depth + 1
        

    def add_children(self, children : List['ActionNode']):
        """Add a child ActionNode to the current node."""
        self.children.extend(children)
    
    def get_children_nodes(self) -> List['ActionNode']:
        """Return a list of child nodes."""
        return self.children

    def __str__(self):
        """Return a string representation of the ActionNode."""
        return f"{self.content} ({ActionNodeState(self.state).name})"
    
    def is_state(self, state: ActionNodeState) -> bool:
        """比较当前状态与给定的状态是否相同。"""
        if isinstance(state, ActionNodeState):
            if isinstance(self.state, ActionNodeState):
                return self.state == state
            elif isinstance(self.state, int):
                return self.state == state.value
            else:
                raise TypeError("node member variable state must be an instance of ActionNodeState or an integer")
        else:
            raise TypeError("state must be an instance of ActionNodeState")
        
    def get_unexecuted_child_node_id(self):
        """ Return the node_id of the first child node that is not executed.
        Returns:
            int: The node_id of the first child node that is not executed.
        """
        for valid_child_node in self.children:
            if valid_child_node.is_state(ActionNodeState.not_executed):
                return valid_child_node.node_id
        return None
    
    def get_unexecuted_child_node(self):
        """ Return the node of the first child node that is not executed.
        Returns:
            ActionNode: The node of the first child node that is not executed.
        """
        for valid_child_node in self.children:
            if valid_child_node.is_state(ActionNodeState.not_executed):
                return valid_child_node
        return None
    
    def get_valid_unexecuted_node(self):
        """ Return the node of the first child node that is not executed.
        Returns:
            ActionNode: The node of the first child node that is not executed.
        """
        for valid_child_node in self.children:
            if valid_child_node.is_state(ActionNodeState.not_executed) or valid_child_node.is_state(ActionNodeState.valid):
                return valid_child_node
        return None
    
    def get_valid_node(self):
        """ Return the node of the first child node that is valid.
        Returns:
            ActionNode: The node of the first child node that is executed.
        """
        for valid_child_node in self.children:
            if valid_child_node.is_state(ActionNodeState.valid):
                return valid_child_node
        return None

from config import explore_exploitation_config

class ActionTree:
    def __init__(self, chain : Optional[Action_Chain] = None):
        """Initialize an empty ActionTree with no nodes."""
        if chain is not None:
            self.agent_id : int = 0
            self.current_node_ptr : int = 0 # 指向头结点
            self.parent_node_ptr = self.current_node_ptr
            self.max_node_id = 0
            self.max_block_id = 0
            self.nodes: List[ActionNode]  = []

            self.head_node = ActionNode(node_id=self.max_node_id, block_id= self.max_block_id,node_type=ActionNodeType.head_node, state=ActionNodeState.head_node,content="HEAD_NODE", effect="HEAD_NODE",intention ="HEAD_NODE")
            self.nodes.append(self.head_node) # head_node is the first node
            self.max_node_id += 1
            self.max_block_id += 1

            head_node = self.head_node
            # _action_chain = [step.replace("\'","").replace("\"","") for step in chain.split(',')]
            for step in chain.chain: # for step in chain.chain:
                node = ActionNode(node_id=self.max_node_id, block_id= self.max_block_id,
                                            node_type=ActionNodeType.unknown, state=ActionNodeState.not_executed,
                                            content=step.step, effect="", intention = "") # TODO fill it with intention
                self.nodes.append(node)
                head_node.add_child(node)
                head_node = node 
                self.max_node_id += 1
            self.max_block_id += 1
        else:
            self.head_node : ActionNode = None

    # -------------- Validation -------------------
    def evaluate_explore_exploitation(self) -> bool:
        """ evaluate explore , False for keep execute, True for replan. """

        def is_explore(action : str) -> bool:
            action = action.lower()
            if "go to" in action or "explore" in action: # or "transport" in action  transport is
                return True
            else:
                return False

        log_system.PL(""" evaluate explore , True for keep execute, False for replan. """)
        epsilon = explore_exploitation_config["threshold"]
        window_size = explore_exploitation_config["window_size"]
        # Exploration vs. Exploitation
        current_node = self.nodes[self.current_node_ptr]
        next_node = current_node.get_unexecuted_child_node()
        if next_node is None:
            return True

        executed_action_chain = self.get_history_action_chain()
        executed_action_chain.append(next_node.content)

        executed_action_chain = executed_action_chain[-window_size:]
        log_system.PL("---plan---\n",executed_action_chain)

        explore_cnt = 0
        exploitation_cnt = 0
        for action_content in executed_action_chain:
            if is_explore(action_content):
                explore_cnt += 1
            else:
                exploitation_cnt += 1

        if explore_cnt > epsilon:
            return True
        return False
    


    # -------------- Create -------------------
    
    def add_new_chain(self, chain: Action_Chain):
        # 获取当前节点
        head_node = self.nodes[self.current_node_ptr]

        # _action_chain = [step.replace("\'","").replace("\"","") for step in chain.split(',')]
        # 遍历链中的每个步骤
        for step in chain.chain:    
            # 创建一个新的节点
            node = ActionNode(node_id=self.max_node_id, block_id= self.max_block_id,
                                        node_type=ActionNodeType.unknown, state=ActionNodeState.not_executed,
                                        content=step.step, effect="",intention="")
            # 将新节点添加到节点列表中
            self.nodes.append(node)
            # 将新节点添加为当前节点的子节点
            head_node.add_child(node)
            # 更新当前节点为新节点
            head_node = node 
            # 更新节点ID
            self.max_node_id += 1
        # 更新区块ID
        self.max_block_id += 1


    #  -------------- Read -------------------

    def get_next_action(self):
        """
        Return the next ActionNode to be executed.
        If the node.children  is empty, return None
        """
        node = self.nodes[self.current_node_ptr]
        child = node.get_unexecuted_child_node()
        # If the node.children is empty, return None
        if child is not None:
            return child.content   # get the step for Action
        else:
            return None

    
    def next(self):
        """ 将指针向下移动 """
        node = self.nodes[self.current_node_ptr]
        if self.current_node_ptr == len(self.nodes):
            log_system.PL("End of the chain")
        _temp_node = node.get_unexecuted_child_node_id()
        if _temp_node is None:
            return None
        self.current_node_ptr = _temp_node

        # 已经执行过了
        self.nodes[self.current_node_ptr].state = ActionNodeState.valid

        self.save_action_tree("next")
        log_system.PL(f"current_node_ptr: {self.current_node_ptr}")
        
        
    def get_current_action_chain(self, target_block_id: int) -> str:
        """
        Return the action chain, which ready to revise the block.
        """
        log_system.PL(f"search target block id: {target_block_id}")
            
        ans = []

        head_node = self.head_node
        if head_node is None:
            return ""
    
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break
            if head_node.block_id == target_block_id:
                ans.append(head_node.content)

        return " -> ".join(ans)
    


    

    def get_history_action_chain(self) -> List[str]:
        """
        Return the action chain, executed or the same block. 
        """
        ans = []

        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_node()
            if head_node is None:
                break
            ans.append(head_node.content)
        return ans
    

    def get_block_history_action_chain(self, target_block_id: int) -> Tuple[List[ActionNode],List[ActionNode]]:
        """
        Return the action chain, executed or the same block. 
        """
        history_nodes = []
        current_block_nodes = []
        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break # 先写终止条件
            elif head_node.is_state(ActionNodeState.valid):
                history_nodes.append(head_node.content)
            elif head_node.block_id == target_block_id:
                current_block_nodes.append(head_node.content)
            else:
                break
        return history_nodes, current_block_nodes
    

    def get_overall_action_chain(self) -> Tuple[List[ActionNode],List[ActionNode]]:
        """
        Return the action chain, executed or the same block. 
        """
        history_nodes = []
        future_nodes = []
        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break # 先写终止条件
            elif head_node.is_state(ActionNodeState.valid):
                history_nodes.append(head_node.content)
            else:
                future_nodes.append(head_node.content)

        return history_nodes, future_nodes
    
    def get_history_replan_next_two_block_action_chain(self) -> Tuple[List[str],List[str],List[str]]:
        """
        Return the action chain, executed or the same block. 
        """
        history_nodes = []
        current_block_nodes = []
        next_block_nodes = []
        head_node = self.head_node
        change_times = 1

        target_node_id = self.nodes[self.current_node_ptr].get_unexecuted_child_node_id()
        if target_node_id is None:
            while not (head_node is None):
                head_node = head_node.get_valid_unexecuted_node()
                if head_node is None:
                        break # 先写终止条件
                if head_node.is_state(ActionNodeState.valid):
                    history_nodes.append(head_node.content)
            return history_nodes ,[] ,[]
        
        target_block_id = self.nodes[target_node_id].block_id

        FLAG = True
        while not (head_node is None):
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                    break # 先写终止条件
            if FLAG:
                if head_node.is_state(ActionNodeState.valid):
                    history_nodes.append(head_node.content)
                elif head_node.node_id == target_node_id:
                    current_block_nodes.append(head_node.content)
                    FLAG = False
            else:
                if head_node.block_id != target_block_id:
                    if change_times:
                        change_times -= 1
                        target_block_id = head_node.block_id
                        next_block_nodes.append(head_node.content)
                else:
                    break
        return history_nodes, current_block_nodes, next_block_nodes


    def get_history_next_two_blocks_action_chain(self, target_block_id: int = -1, change_times :int = 1) -> Tuple[List[str],List[str],List[str]]:
        """
        Return the action chain, executed or the same block. 
        """
        history_nodes = []
        current_block_nodes = []
        next_block_nodes = []
        head_node = self.head_node

        if target_block_id == -1:
            target_block_id = self.nodes[self.current_node_ptr].block_id
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break # 先写终止条件
            elif head_node.is_state(ActionNodeState.valid):
                history_nodes.append(head_node.content)
            elif head_node.block_id == target_block_id:
                current_block_nodes.append(head_node.content)
            elif head_node.block_id != target_block_id:
                if change_times:
                    change_times -= 1
                    target_block_id = head_node.block_id
                    next_block_nodes.append(head_node.content)
            else:
                break
        return history_nodes, current_block_nodes, next_block_nodes
    
    def get_history_current_next_action_chain(self, target_block_id: int) -> Tuple[List[str],List[str],List[str]]:
        """
        Return the action chain, executed or the same block. 
        """
        history_nodes = []
        current_nodes = []
        next_nodes = []
        head_node = self.head_node
        change_times = 1
        next_action_node_ptr =  self.nodes[self.current_node_ptr].get_unexecuted_child_node_id()
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break # 先写终止条件
            elif head_node.is_state(ActionNodeState.valid):
                history_nodes.append(head_node.content)
            elif head_node.node_id == next_action_node_ptr: 
                target_block_id = head_node.block_id
                current_nodes.append(head_node.content)
            elif head_node.block_id != target_block_id:
                if change_times:
                    change_times -= 1
                    target_block_id = head_node.block_id
                    next_nodes.append(head_node.content)
            else:
                break
        return history_nodes, current_nodes, next_nodes
    

    def get_block_action_chain(self, target_block_id: int) -> List[str]:
        """ return action chain, executing step, intention of the action chain """
        current_block_nodes = []
        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break
            if head_node.block_id == target_block_id:
                current_block_nodes.append(head_node.content)
            
        return current_block_nodes


    def get_refine_action_info(self, history_length :int = 10) -> str:
        """ return action chain, executing step, intention of the action chain """

        action_chain_info = "" # 返回结果

        target_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        if target_node_ptr is None:
            history_action_chain = self.get_history_action_chain()
            # 根据 history_length 的值调整 recently_action_chain
            if history_length > 0:
                recently_action_chain = history_action_chain[-history_length:]  # 获取最近的 history_length 个动作
            else:
                recently_action_chain = history_action_chain[:-1]  # 不获取任何动作

            if recently_action_chain:
                recently_action_chain_info = ",".join(recently_action_chain)
                action_chain_info += f"Plans that have already been executed: {recently_action_chain_info}.\n "
                action_chain_info += "Plans to be executed: "

        else:
            target_node = self.nodes[target_node_ptr]
            

            # history_action_chain, current_block_action_chain = self.get_block_history_action_chain(target_block_id=target_node.block_id)
            # history_action_chain, current_block_action_chain = self.get_overall_action_chain()
            history_action_chain, current_block_action_chain , next_block_action_chain = self.get_history_next_two_blocks_action_chain(target_node.block_id)
            # 根据 history_length 的值调整 recently_action_chain
            if history_length > 0:
                recently_action_chain = history_action_chain[-history_length:]  # 获取最近的 history_length 个动作
            else:
                recently_action_chain = history_action_chain[:-1]  # 不获取任何动作

            if recently_action_chain:
                recently_action_chain_info = ",".join(recently_action_chain)
                action_chain_info += f"Plans that have already been executed: {recently_action_chain_info}.\n "

            if current_block_action_chain:
                current_block_action_chain = ','.join(current_block_action_chain)
                action_chain_info +=  f"Your newly planned action plan will override this original action plan: {current_block_action_chain}. "
            
            if next_block_action_chain:
                next_block_action_chain = ','.join(next_block_action_chain)
                action_chain_info +=  f" The subsequent action sequence for the newly added planning sequence is: {next_block_action_chain}. "


            # if target_node.content:
            #     action_chain_info +=  f"Next Step is: {target_node.content}. "

            if target_node.intention:
                action_chain_info += f"Here is the intention of the action: {target_node.intention} "

        return action_chain_info
    
    def get_create_action_chain_info(self, history_length :int = 10) -> str:
        """ return action chain, executing step, intention of the action chain """

        action_chain_info = "" # 返回结果

        target_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        if target_node_ptr is None:
            history_action_chain = self.get_history_action_chain()
            # 根据 history_length 的值调整 recently_action_chain
            if history_length > 0:
                recently_action_chain = history_action_chain[-history_length:]  # 获取最近的 history_length 个动作
            else:
                recently_action_chain = history_action_chain[:-1]  # 不获取任何动作

            if recently_action_chain:
                recently_action_chain_info = ",".join(recently_action_chain)
                action_chain_info += f"Plans that have already been executed: {recently_action_chain_info}.\n "
                action_chain_info += "There is no plan to execute. "

        else:
            target_node = self.nodes[target_node_ptr]
            


            history_action_chain, current_block_action_chain , next_block_action_chain = self.get_history_replan_next_two_block_action_chain()
            # 根据 history_length 的值调整 recently_action_chain
            if history_length > 0:
                recently_action_chain = history_action_chain[-history_length:]  # 获取最近的 history_length 个动作
            else:
                recently_action_chain = history_action_chain[:-1]  # 不获取任何动作

            if recently_action_chain:
                recently_action_chain_info = ",".join(recently_action_chain)
                action_chain_info += f"Executed plans: {recently_action_chain_info}.\n "

            if target_node.content:
                action_chain_info +=  f"Your newly planned action plan will override this original action : {target_node.content}. "
            
            if next_block_action_chain:
                next_block_action_chain = ','.join(next_block_action_chain)
                action_chain_info +=  f"The subsequent action sequence for the newly added planning sequence is: {next_block_action_chain}. "



            if target_node.intention:
                action_chain_info += f"Here is the intention of the action: {target_node.intention} "

        return action_chain_info
    

    def get_refine_action_step_info(self, history_length :int = 10) -> str:
        """ return action chain, executing step, intention of the action chain """

        action_chain_info = "" # 返回结果

        target_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        if target_node_ptr is None:
            history_action_chain = self.get_history_action_chain()
            # 根据 history_length 的值调整 recently_action_chain
            if history_length > 0:
                recently_action_chain = history_action_chain[-history_length:]  # 获取最近的 history_length 个动作
            else:
                recently_action_chain = history_action_chain[:-1]  # 不获取任何动作

            if recently_action_chain:
                recently_action_chain_info = ",".join(recently_action_chain)
                action_chain_info += f"Plans that have already been executed: {recently_action_chain_info}.\n "
        else:
            target_node = self.nodes[target_node_ptr]
            

            # history_action_chain, current_block_action_chain = self.get_block_history_action_chain(target_block_id=target_node.block_id)
            # history_action_chain, current_block_action_chain = self.get_overall_action_chain()
            history_action_chain, current_block_action_chain , next_block_action_chain = self.get_history_current_next_action_chain(target_node.block_id)
            # 根据 history_length 的值调整 recently_action_chain
            if history_length > 0:
                recently_action_chain = history_action_chain[-history_length:]  # 获取最近的 history_length 个动作
            else:
                recently_action_chain = history_action_chain[:-1]  # 不获取任何动作

            if recently_action_chain:
                recently_action_chain_info = ",".join(recently_action_chain)
                action_chain_info += f"Plans that have already been executed: {recently_action_chain_info}.\n "

            if current_block_action_chain:
                current_block_action_chain = ','.join(current_block_action_chain)
                action_chain_info +=  f"    Your newly planned action plan will override this original action plan: {current_block_action_chain}. "
            
            if next_block_action_chain:
                next_block_action_chain = ','.join(next_block_action_chain)
                action_chain_info +=  f"The subsequent action sequence for the newly added planning sequence is: {next_block_action_chain}. "


            # if target_node.content:
            #     action_chain_info +=  f"Next Step is: {target_node.content}. "

            if target_node.intention:
                action_chain_info += f"Here is the intention of the action: {target_node.intention} "

        return action_chain_info
    
    #  -------------- Display -------------------
    
    def get_current_action_block_info(self, block_id :  Optional[int] = None) -> str:
        """ return action chain, executing step, intention of the action chain """
        target_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        if target_node_ptr is not None:
            target_node = self.nodes[target_node_ptr]
            if block_id is None:
                block_id = target_node.block_id
                log_system.PL(f"block_id: {block_id}")
            action_chain = self.get_current_action_chain(block_id)        
            action_chain_info = ""

            if action_chain:
                action_chain_info += f"Here is the action chain: {action_chain}. "

            if target_node.content:
                action_chain_info +=  f"Going to execute the step: {target_node.content}. "

            if target_node.intention:
                action_chain_info += f"Here is the intention of the action chain: {target_node.intention}. "

            return action_chain_info
        return "There is no plan."
    
    def display_action_chain_with_block_info(self) -> str:
        """ return action chain and each step block id """
        overall_chain = []
        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break # 先写终止条件
            elif head_node.node_id == self.current_node_ptr:
                overall_chain.append(f"###({head_node.content}) (block: {head_node.block_id})###")
            else:
                overall_chain.append(f"({head_node.content}) (block: {head_node.block_id})")
        return " -> ".join(overall_chain)

    #  -------------- Display -------------------

    
                
    def display_overall_action_tree(self, block_id :  Optional[int] = None, flag_log : bool = True):
        """ 展示以及录入日志 """
        target_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        if target_node_ptr is not None:
            target_node = self.nodes[target_node_ptr]
            if block_id is None:
                block_id = target_node.block_id
            action_chain_text = self.get_overall_plan_in_text()        
            action_chain_info = f"Here is the action chain: {action_chain_text}. Going to execute the step {target_node.content}. Here is the intention of the action chain: {target_node.intention}."
            if flag_log:
                log_system.PL(action_chain_info)
            else:
                log_system.PRINT(action_chain_info)
        else:
            return "There is no plan."

    def retrieve_nodes_by_content(self, action_content : str) -> Tuple[Union[int,ActionNodeState],List[str]]:
        """
        Retrieve all nodes with the specified content.
        # 数字大小表示覆盖顺序（优先级）
        head_node = 0
        # 动作节点未执行
        not_executed = 1
        # 动作节点无效
        invalid = 2
        # 动作结点已经执行过（ 给 replan 专用 )
        executed = 3
        # 动作节点有效
        valid = 4
        return 
        """

        flag = -1 # 最小值
        
        current_node = self.nodes[self.current_node_ptr]
        current_depth = current_node.depth
        # visited_block_id = set()
        conflict_messages : list[str] = []

        
        log_system.PL(f"Searching for nodes with content: {action_content}")

        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break
            elif abs(current_depth - head_node.depth) < 7 and action_content in head_node.content:
                flag = node_state_max(flag,head_node.state)
                # action_chain = self.get_refine_action_step_info(node.block_id)
                block_action_chain_message = "The former agent's next step :" +  action_content + " is overlap with the the latter agent."
                conflict_messages.append(block_action_chain_message)
                log_system.LOG(f"target node conflict with {head_node.content} (block: {head_node.depth})")
               
        if len(conflict_messages) == 0:
            log_system.LOG(f"No nodes found with content: {action_content}")
        else:
            log_system.LOG(f"conflict_messages : \n {conflict_messages}")        

        return flag, conflict_messages


    def get_overall_plan_in_text(self) -> str:
        """ bfs the valid, not_executed nodes. Each level only contains one node. """

        ans = []

        head_node = self.head_node
        while True:
            head_node = head_node.get_valid_unexecuted_node()
            if head_node is None:
                break
            ans.append(f"||{head_node.content}||" if head_node.node_id == self.current_node_ptr else head_node.content)
        
        result = [step for step in ans]

        return " -> ".join(result)
    

    # -------------- Update -------------------

    def set_current_node_invalid(self):
        self.nodes[self.current_node_ptr].state = ActionNodeState.invalid



    # -------------- Modification -------------------
        
    def skip_node(self):
        """ Skip an ActionNode from the ActionTree. """
        # 直接跳过 节点 node_id 用于 步骤失效
        head_node = self.nodes[self.current_node_ptr]
        skip_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)

        end_node_ptr = None
        if skip_node_ptr is not None:    
            skip_node = self.nodes[skip_node_ptr]
            skip_node.state = ActionNodeState.invalid
            end_node_ptr = self.get_next_unexecuted_node_id(skip_node_ptr)



        if end_node_ptr is not None:
            end_node = self.nodes[end_node_ptr]
            head_node.add_child(end_node)
        
        self.save_action_tree("skip node")

    def get_next_unexecuted_node_id(self, node_id : int):
        if node_id is None:
            return None
        return self.nodes[node_id].get_unexecuted_child_node_id()


    def refine_node(self, single_step : Action):
        """ 
        Refine 节点 self.current_node_ptr and create a new node for it.
        """
        head_node = self.nodes[self.current_node_ptr]
        refine_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        _block_id = 0
        end_node_ptr = None
        _intention = ""
        if refine_node_ptr is not None:
            end_node_ptr = self.get_next_unexecuted_node_id(refine_node_ptr)
            self.nodes[refine_node_ptr].state = ActionNodeState.invalid
            refine_node = self.nodes[refine_node_ptr]
            _block_id = refine_node.block_id
            _intention = refine_node.intention
        else:
            _block_id = self.max_block_id
            self.max_block_id += 1

        # 接上
        new_node = ActionNode(
            node_id=self.max_node_id,
            block_id=_block_id,
            node_type=ActionNodeType.unknown,
            state=ActionNodeState.not_executed,
            content=single_step.step,
            effect="",intention=_intention) # TODO Maybe need to provide a reason here.
        self.nodes.append(new_node)
        self.max_node_id += 1


        head_node.add_child(new_node)

        if end_node_ptr is not None:
            end_node = self.nodes[end_node_ptr]
            new_node.add_child(end_node)
            refine_node.children = []

        self.save_action_tree("refine node") 
    
    def replace_node(self, action_content : str):
        """ 
        Refine 节点 self.current_node_ptr and create a new node for it.
        """
        head_node = self.nodes[self.current_node_ptr]
        refine_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)
        end_node_ptr = self.get_next_unexecuted_node_id(refine_node_ptr)

        self.nodes[refine_node_ptr].state = ActionNodeState.invalid
        refine_node = self.nodes[refine_node_ptr]

        # 接上
        new_node = ActionNode(
            node_id=self.max_node_id,
            block_id=refine_node.block_id,
            node_type=ActionNodeType.unknown,
            state=ActionNodeState.not_executed,
            content=action_content,
            effect="",intention=refine_node.intention) # TODO Maybe need to provide a reason here.
        self.nodes.append(new_node)
        self.max_node_id += 1


        head_node.add_child(new_node)

        if end_node_ptr is not None:
            end_node = self.nodes[end_node_ptr]
            new_node.add_child(end_node)
            refine_node.children = []

        self.save_action_tree("replace node")    

    def refine_action_chain(self, action_chain : Action_Chain):
        """ 直接取代原来的整个block的action """
        start_node = self.nodes[self.current_node_ptr]
        head_node = start_node.get_unexecuted_child_node()
        # 将之前的结点转为invalid
        if head_node is None:
            self.add_new_chain(action_chain)
        else:
            head_node.state = ActionNodeState.invalid
            target_block_id = head_node.block_id 
            current_block_id = target_block_id

            child_node_id = head_node.get_unexecuted_child_node_id()

            end_node = None
            while True: 
                if child_node_id is None:
                    break
                end_node = self.nodes[child_node_id]
                current_block_id = end_node.block_id
                if current_block_id == target_block_id:
                    end_node.state = ActionNodeState.invalid
                else:
                    break
                child_node_id = end_node.get_unexecuted_child_node_id()
            
            if end_node is None:
                end_node = head_node

            new_head_node = start_node
            
            # block_id = child_node_block_id
            for step in action_chain.chain:
                # 创建一个新的节点
                node = ActionNode(node_id=self.max_node_id, block_id= target_block_id,
                                            node_type=ActionNodeType.unknown, state=ActionNodeState.not_executed,
                                            content=step.step, effect="",intention=action_chain.intention)
                # 将新节点添加到节点列表中
                self.nodes.append(node)
                # 将新节点添加为当前节点的子节点
                new_head_node.add_child(node)
                # 更新当前节点为新节点
                new_head_node = node 
                # 更新节点ID
                self.max_node_id += 1

            if child_node_id is not None:
                # clear 
                child_node = self.nodes[child_node_id]
                new_head_node.add_child(child_node)
                end_node.children = []

        self.save_action_tree()


    def merge_action_chain(self, task_chain: Action_Chain):
        """Replace a chain of actions (tasks) into the specified ActionNode."""
        replan_node_ptr = self.get_next_unexecuted_node_id(self.current_node_ptr)

        if replan_node_ptr is None:
            return None
        child_node_ptr = self.get_next_unexecuted_node_id(replan_node_ptr)
        
        self.nodes[replan_node_ptr].state = ActionNodeState.executed # stop the chain    
        head_node = self.nodes[self.current_node_ptr]
        # 接上
        for step in task_chain.chain:    
            node = ActionNode(node_id=self.max_node_id, block_id= self.max_block_id,
                                        node_type=ActionNodeType.unknown, state=ActionNodeState.not_executed,
                                        content=step.step, effect="",intention = task_chain.intention)

            self.max_node_id += 1
            self.nodes.append(node)
            head_node.add_child(node)
            head_node = node
        self.max_block_id += 1

        if child_node_ptr is not None:
            head_node.add_child(self.nodes[child_node_ptr])
            self.nodes[replan_node_ptr].children = []

        self.save_action_tree()
    


    def visualize_tree(self, node: Optional[ActionNode] = None, level: int = 0) -> None:
        """Print the structure of the ActionTree in a visual format."""
        if node is None:
            for root in self.nodes:
                print("  " * level + str(root))
                self.visualize_tree(root, level + 1)
        else:
            for child in node.get_children_nodes():
                print("  " * (level + 1) + str(child))
                self.visualize_tree(child, level + 1)

    # save

    def save_tree_to_json(self, file_path: str = "tree.json"):
        """Save the ActionTree to a JSON file."""
        tree_data = [node.to_dict() for node in self.nodes]
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(tree_data, json_file, ensure_ascii=False, indent=4)

    def save_action_tree(self, explanation : str = ""):
        """ Save the action tree for log """
        log_system.log_planning_data(f"Agent {self.agent_id} : {self.get_overall_plan_in_text()} : {explanation}")

    def get_next_node_depth(self, node_id: int) -> int:
        """
        Get depth of the next unexecuted node 
        """
        if node_id is None:
            node_id = self.current_node_ptr
        child_node = self.get_next_unexecuted_node_id(node_id)
        if child_node is None:
            return -1
        else:
            return child_node.depth
    
    def get_current_node_depth(self) -> int:
        """
        Get depth of the current node 
        """
        return self.nodes[self.current_node_ptr].depth


def check_node_similarity(node1: ActionNode, node2: ActionNode) -> bool:
    """Check if two ActionNodes are similar based on their content."""
    return node1.content in node2.content or node2.content in node1.content


def get_similarity(node1: ActionNode, node2: ActionNode) -> int:
    """Calculate and return the number of similar words in the content of two ActionNodes."""
    return len(set(node1.content.split()).intersection(set(node2.content.split())))

def validate_tree(node : ActionNode, tree_b_head_node: ActionNode):
    """Validate the current tree against another tree based on node similarity."""

    queue = deque([tree_b_head_node])  # Initialize the queue with the head node
    flag_end = False
    parent_node_ = tree_b_head_node
    child_node_ = None
    while queue and not flag_end:
        current_node = queue.popleft()  # Dequeue the front node
        for child_node in current_node.get_children_nodes():
            if check_node_similarity(node,child_node):
                flag_end = True
                parent_node_ = current_node
                child_node_ = child_node
                break
            for child in current_node.get_children_nodes():
                queue.append(child)
    return child_node_.state



# -------------------------------- Action Tree End -------------------------------------------

# ------------------------------智能體設置 START----------------------------


from config import MAX_PLANNING_TIMES

def get_goals(text: str):
    # 使用正则表达式匹配数字后面的名词
    matches = re.findall(r'(\d+)\s+([a-zA-Z]+)', text)
    items_dict = {match[1]: int(match[0]) for match in matches}
    print("get_goals :", items_dict)
    return items_dict



class MemoryGraph():
    def __init__(self):
        self.rooms = {}  # 存储房间节点
        # observation of objects
        self.objects = {} # 存储物体节点
        self.room_edges = {}  # 存储房间之间的边及其权重
        self.holdings = dict() # agent id -> (object_id, object_name, agent_name)
        self.observation :dict = dict() # agent 作为特殊的room

    # 添加房间
    def add_room(self, room_id, room):
        # 将房间id和房间信息存入字典中
        self.rooms[room_id] = {'id': room_id, 'room': room}

    def delete_object(self, target_object_id):
        target_object = self.objects.get(target_object_id)
        if not (target_object is None):
            self.objects
                
    def update_observation(self,room_id ,descriptions:List[str]):
        new_descriptions = []
        
        for description in descriptions:
            if ',' in description:
                for new_description in description.split(','):
                    new_descriptions.append(new_description.strip()) 
            else:
                new_descriptions.append(description)
                    
        if room_id not in self.observation:
            self.observation[room_id] = new_descriptions
        else:
            self.observation[room_id].extend(new_descriptions)

    def reset_observation(self):
        self.observation = dict()
        
    def update_holding(self, holding_description : str , agent_id, object_id, object_name):
        _room = None
        _index = None
        target_object = f"<{object_name}> ({object_id})"
        for room , descriptions in self.observation.items():
            for index, description in enumerate(descriptions):
                if target_object in description:
                    _room = room
                    _index = index
                    break
        if _room and _index:
            del self.observation[_room][_index]

        self.holdings[agent_id] = holding_description


    def get_relevant_obj_info(self, obj :str):
        """ get which room the object is """
        # print("get_relevant_obj_info object_id",obj)
        for room , descriptions in self.observation.items():
            for description in descriptions:
                if obj in description:
                    return room
        return ""
    
    def add_room_edge(self, room1_id, room2_id):
        if (room1_id, room2_id) not in self.room_edges:
            self.room_edges[(room1_id, room2_id)] = None  # 初始化权重为 None

    def set_room_weight(self, room1_id, room2_id, cost):
        if (room1_id, room2_id) not in self.room_edges:
            self.room_edges[(room1_id, room2_id)] = None  # 初始化权重为 None
            self.room_edges[(room2_id, room1_id)] = None  # 初始化权重为 None

        if (room1_id, room2_id) in self.room_edges:
            self.room_edges[(room1_id, room2_id)] = cost
            self.room_edges[(room2_id, room1_id)] = cost  # 无向图
    
    def get_room_direct_cost(self, room1_id, room2_id):
        return self.room_edges.get((room1_id, room2_id), None)
    
    # ------------------------------------------------------------
    # 
    # ------------------------------------------------------------

    def change_object_location(self, object_id, new_room_id):
        if object_id in self.objects:
            self.objects[object_id]['location'] = new_room_id

    def get_room_neighbors(self, room_id):
        return [room_id2 for (room_id1, room_id2) in self.room_edges if room_id1 == room_id]

    def query_cost_of_moving(self, start_room_id, end_room_id):
        """ Current use simplest function to evalute the cost. """
        neighbors = self.get_room_neighbors(start_room_id)
        if neighbors:
            weights = {neighbor: self.room_edges.get((end_room_id, neighbor)) for neighbor in neighbors}
        else:
            neighbors = None
            weights = None
        return neighbors, weights
    




class Memory():
    def __init__(self):
        self.memory_graph = MemoryGraph()
        self.last_room = {}

    def update_room_cost(self, msg : PROMPT_Constituent_Elements):
        agent_id = get_name_to_id(msg.agent_name)
        if self.last_room.get(agent_id) is None:
            self.last_room[agent_id] = (msg.current_room, msg.step_num)
        elif self.last_room[agent_id][0] != msg.current_room:
            cost = msg.step_num - self.last_room[agent_id][1]
            self.memory_graph.set_room_weight(self.last_room[agent_id][0], msg.current_room, cost)
            self.last_room[agent_id] = (msg.current_room, msg.step_num)


    def update_observation(self, msg : PROMPT_Constituent_Elements):
        self.memory_graph.reset_observation()
        for room, descriptions in msg.discovery.items():
            self.memory_graph.update_observation(room,descriptions)

    # def update_holding(self, msg : PROMPT_Constituent_Elements):
    #     agent_id = get_name_to_id(msg.agent_name)
    #     holding_description, t = self.construct_holding_description(msg.grabbed_objects)
    #     for object_id , object_name in t.items():
    #         self.memory_graph.update_holding(holding_description,agent_id,object_id,object_name)

    def display_overall_observation(self):
        pprint(self.memory_graph.observation)


    def get_observation_text(self):
        return f"Observation from memory_graph:\n {self.memory_graph.observation}"

    def construct_holding_description(self, grabbed_objects : List) -> Tuple[str, Dict[int, str]]:
        t = dict()
        s = ""
        if len(grabbed_objects) == 0:
            s += "It's holding nothing. "
        else:
            s += f"It's holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            t[grabbed_objects[0]['id']] =  grabbed_objects[0]['class_name']

            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
                t[grabbed_objects[1]['id']] =  grabbed_objects[1]['class_name']

            # if len(grabbed_objects) == 2:
                # s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
                # t[grabbed_objects[1]['id']] =  grabbed_objects[1]['class_name']
                # holding[grabbed_objects[1]['id']] = grabbed_objects[1]['class_name']
        return s, t
    
    def update_on_msg(self, msg : PROMPT_Constituent_Elements):
        self.update_room_cost(msg)
        self.update_observation(msg)
        # self.update_holding(msg)

    def retrieve_relevant_info(self, agent_id : int, option : str):
        """ retrieve cost of moving between two rooms or relevant info"""
        
        location = extract_pairs(option) # location:  [('165', 'fridge')]
        log_system.PRINT("location: ",location) 
        if is_location(location[0][1]):
            current_room = extract_pairs(self.last_room[agent_id][0])
            log_system.PRINT("current_room: ",current_room)

                
            
    def get_task_relevant_info(self,task_content : str):
        goals = get_goals(task_content)
        relevant_info = ""
        for goal , number in goals.items():
            temp_info = self.memory_graph.get_relevant_obj_info(goal)
            relevant_info += temp_info
        return relevant_info


    def get_available_actions_relevant_info(self, agent_id : int, available_actions : str):
        """ retrieve relevant info for available actions """
        cleaned_available_plans = remove_labels(available_actions)
        options = [option.lower() for option in cleaned_available_plans.split("\n") if option.strip() and not "send a message" in option ]
        relevant_obj_info = dict()
        relevant_moving_info = ""
        for option in options:
            location = extract_pairs(option) # location:  [('165', 'fridge')]
            if location == []:
                pass
            elif is_location(location[0][1]):
                current_room = extract_pairs(self.last_room[agent_id][0])
                cost = self.memory_graph.get_room_direct_cost(current_room[0][0], location[0][0])
                if not (cost is None):
                    relevant_moving_info += f"Need {cost} steps from {location[0][1]} to <{location[0][1]}> ({location[0][0]})"
            else:
                object_name_id = f"<{location[0][0]}> ({location[0][1]})"
                room = self.memory_graph.get_relevant_obj_info(object_name_id)
                info = (room,object_name_id)
                if info[0] != "":
                    if info[0] in  relevant_obj_info:
                        relevant_obj_info[info[0]].append(info[1])
                    else:
                        relevant_obj_info[info[0]] = [info[1]]
        ans = relevant_moving_info
        if relevant_obj_info != {}:
            ans = f"Agent {agent_id} can see "
            for room, objects in  relevant_obj_info.items():
                ans += f"{room} contains {','.join(objects)}. "
        return ans
         

class Agent():
    def __init__(self, /, **data) -> None:
        
        self.agent_id: int = data["agent_id"]
        self.state: AGENT_STATE = get_initial_agent_state()
        self.action: Optional[Agent_Decision] = None
        self.stage: Stage = Stage.initial_dispatching_task
        
        # for log
        self.steps_data: List = []
        self.current_step_data: List = []
        
        # for replan times
        self.planning_counter: int = 0
        self.max_planning_times: int = MAX_PLANNING_TIMES
        
        self.action_tree: ActionTree = ActionTree()
        # disposable planning reason 
        self.disposable_planning_reason: Optional[str] = None
        self.exemption_list : Dict = dict() 
        self.exemption_valid_time : int = k_exemption_valid_time # 1 则当前轮是有效的

        # Newest msg
        self.msg : PROMPT_Constituent_Elements = PROMPT_Constituent_Elements()


    def next(self):
        self.state['need_replan'] = False
        self.action_tree.next()
        for key in self.exemption_list.keys():
            if self.exemption_list[key] > 0:
                self.exemption_list[key] -= 1       

    def check_exemption_list(self, action_content : str):
        """ 放入豁免名单并刷新对应的豁免时长，如果是刚刚添加进去的，就返回False， 不要重复处理这个冲突，否则返回True,处理这个冲突 """
        if "goput" in action_content: # 强制不让goput引发冲突
            return False
        exemption = self.exemption_list.get(action_content)
        if exemption is None :
            self.exemption_list[action_content] = k_exemption_valid_time
            return True
        elif exemption == 0:
            self.exemption_list[action_content] = k_exemption_valid_time
            return True
        elif exemption < k_exemption_valid_time:
            self.exemption_list[action_content] = k_exemption_valid_time
            return False
        else: # exemption == k_exemption_valid_time
            return False

    def set_action_tree(self, action_chain : Action_Chain):
        self.action_tree = ActionTree(action_chain)
        self.action_tree.agent_id = self.agent_id
    
    def is_reach_limit(self) -> bool:
        """
        True 超出规划限定
        False  没有超出规划限定
        """
        ans_flag = self.planning_counter == self.max_planning_times
        self.planning_counter += 1
        return ans_flag
    
    def is_exceeding_limit(self) -> bool:
        """
        True 超出规划限定
        False  没有超出规划限定
        """
        return self.planning_counter > self.max_planning_times
    
    

    def initial_planning_counter(self):
        self.planning_counter = 0


    def set_disposable_planning_reason(self, reason: str):
        self.disposable_planning_reason = reason

    def get_disposable_planning_reason(self) -> Optional[str]:
        temp = self.disposable_planning_reason 
        self.disposable_planning_reason = ""
        return temp
    
    
    def display_action_chain_with_block_info(self) -> str:
        info = self.action_tree.display_action_chain_with_block_info()
        print(info)
        return info

        


def SO_to_string(agent_state_dict: AGENT_STATE) -> str:
    ans = []
    if agent_state_dict["observation"]:
        ans.append(agent_state_dict["observation"])
    if agent_state_dict["state"]:
        ans.append(agent_state_dict["state"])
    return "\n".join(ans)


STEP_WINDOW_SIZE = 5


def get_current_step(agent_state_dict: AGENT_STATE) -> Union[Action, None]:
    return agent_state_dict["current_cot_step"]


# ------------------------------智能體數據結構 END----------------------------




def update_state(agent: Agent,prompt_elements:PROMPT_Constituent_Elements):

    # 直接更新選擇題

    # 从 prompt_elements 中提取信息再更新對應的文本描述
    # agent_name = prompt_elements.agent_name
    # current_room = prompt_elements.current_room
    # rooms_explored = prompt_elements.rooms_explored
    # holding_objects = prompt_elements.holding_objects
    # obj_per_room = prompt_elements.obj_per_room
    # opponent_grabbed_objects = prompt_elements.opponent_grabbed_objects
    progress_desc = prompt_elements.progress_desc
    # current_step = prompt_elements.step_num
    # goal_desc = prompt_elements.goal_desc
    # agent.state['available_plans'] = available_plans

    agent.state['observation'] = progress_desc + "Action space of this agent is:\n" + prompt_elements.available_plans
    agent.state['need_replan'] = prompt_elements.need_replan
    # 为了避免麻烦，我直接将这个目标放到状态这里了
    # agent.state['current_task'] = 
    agent.state['state'] = (
        f"The overall goal is : {prompt_elements.current_task} "
    )
    agent.state['current_task'] = prompt_elements.current_task
    agent.msg = prompt_elements
    # log_system.PRINT("------------------agent.state['observation'] ------------------\n",agent.state['observation'])
    # log_system.PRINT("------------------agent.state['state'] ------------------\n",agent.state['state'])

    agent.state['available_plans'] = prompt_elements.available_plans


    return agent

def get_OS(agent: Agent) -> str:
    return agent.state['observation'] + agent.state['state']
        

def set_all_agents_stage(agents :list[Agent],target_stage : Stage):
    for agent in agents:
        agent.stage =  target_stage



# ------------------------------智能體數據結構 END----------------------------





# ---------------------------- 通用函数 START ------------------------

def count_cost(input_tokens:int, output_tokens:int):
    from config import model_name
    if model_name == 'gpt-4':
        return input_tokens* 30 / 1000000 + output_tokens * 60 / 1000000
    elif model_name == 'gpt-4o':
        return input_tokens * 5 / 1000000 + output_tokens * 15 / 1000000
    elif model_name == 'gpt-4o-mini':
        return input_tokens* 0.15   / 1000000 + output_tokens * 0.6 / 1000000
    else:
        raise Exception("Unknown model name")


class USAGE(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float

import random
from config import max_request_times, k_exemption_valid_time



def clean_json_output(json_string : str) -> str:
    cleaned_json = re.sub(r'//.*', '', json_string)
    cleaned_json = cleaned_json.replace('\n', '')
    return cleaned_json


async def wrap_invoke(prompt:str,source_name : str, wrap_function):
    """ 为了带上日志系统，所以包装一下调用函数 """
# 纯粹为了规范这样写这个代码
    ans = None 
    usage = None 
    if "gpt" in model_name:
        for i in range(max_request_times):
            try:
                log_system.PRINT(" 向大模型服务器发送请求 ")
                start_time = time.perf_counter()  # 使用高精度计时器
                prompt = prompt.replace("..",".")
                # log_system.PRINT(prompt)
                response = GLOBAL_LLM.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}])

                elapsed_time = time.perf_counter() - start_time  # 计算耗时
                
                ans = response.choices[0].message.content
                log_system.PL("response:\n",ans)
                ans = clean_json_output(ans)
                ans = wrap_function(ans)
                log_system.PRINT("wrap_function_result:\n",ans)

                elapsed_time_rounded = round(elapsed_time, 6)  # 四舍五入到六位小数
                
                log_system.PRINT(f"Elapsed Time: {elapsed_time_rounded} seconds")  # 打印耗时，保留六位小数
                prompt_token_usage = response.usage.prompt_tokens
                completion_token_usage = response.usage.completion_tokens
                total_token_usage = prompt_token_usage + completion_token_usage
                usage = USAGE(prompt_tokens=prompt_token_usage,
                               completion_tokens=completion_token_usage, 
                               total_tokens=total_token_usage,
                            total_cost=count_cost(prompt_token_usage,completion_token_usage))
                
                
                if ans:
                    log_system.refine_call_counts(source_name)
                    await log_system.log_token_usage(usage,source_name,str(ans),elapsed_time_rounded)
                    break
                else:
                    log_system.PRINT("Error : ans is None")

            except Exception as e: # requests.exceptions.RequestException as e:
                log_system.PRINT(f"Error: {e}")
                special_num = 60 + random.randint(2, 4)
                additional_message = f"Just pick an action in the actions can be performmed. Think carefully. You are losing your chance: ({i+1}/{max_request_times})"
                await asyncio.sleep(special_num) 
        return ans
    
    else: #  "qwen2.5-7B" in model_name:
        while ans is None:
            log_system.PRINT(" 向大模型服务器发送请求 ")
            prompt = prompt.replace("..",".")

            log_system.PL(f"\n{prompt}")
            response = GLOBAL_LLM.invoke(prompt)
            
            ans = response["response"]
            log_system.PL("\nresponse:\n",ans)
            ans = clean_json_output(ans)
            ans = wrap_function(ans)

            if ans is None:
                log_system.PRINT("Error : ans is None")
                special_num = 5 + random.randint(2, 4)
                await asyncio.sleep(special_num)
                continue

            # log_system.PRINT("wrap_function_result:\n",ans)
            elapsed_time_rounded = response["time"]
            
            log_system.PRINT(f"Elapsed Time: {elapsed_time_rounded} seconds")  # 打印耗时，保留六位小数

            usage = USAGE(prompt_tokens=response['input_token_count'], completion_tokens=response['output_token_count'], total_tokens=response['output_token_count'] + response["input_token_count"], total_cost=0)
            await log_system.log_token_usage(usage,source_name,str(ans),elapsed_time_rounded)

            log_system.refine_call_counts(source_name)
        return ans



from config import tot_search_size


async def generate_multiple_request(llm_with_structure, prompt:str, source_name:str):
    tasks = [asyncio.create_task(wrap_invoke(llm_with_structure, prompt, source_name)) for _ in range(tot_search_size)]
    results = await asyncio.gather(*tasks) 
    return results

    

class frame_saved_path(BaseModel):
    path : str
    success_number : int


class StateMachineState(Enum):
    dispatching_i = 1
    dispatching = 2
    generate_action_sequence_i = 3
    generate_action_sequence = 4
    get_action = 5
    attempt_execution = 6

    finish_state = 7

def state_min(state0: Optional[Union[StateMachineState, int]], state1: Optional[Union[StateMachineState, int]]) -> StateMachineState:
    # 提取状态值，考虑可能为 None 的情况
    value0 = state0.value if isinstance(state0, StateMachineState) else state0
    value1 = state1.value if isinstance(state1, StateMachineState) else state1
    # 处理 None 值
    if value0 is None:
        return StateMachineState(value1)
    if value1 is None:
        return StateMachineState(value0)

    # 使用枚举值进行比较
    _next_state_value = min(value0, value1)

    # 转换回 StateMachineState
    _next_state = StateMachineState(_next_state_value)
    return _next_state

def convert_to_batch_prompt(data: dict) -> BATCH_PROMPT_Constituent_Elements:
    return BATCH_PROMPT_Constituent_Elements(**data)


def construct_holding_description(grabbed_objects : Union[List,str]) -> str:
    if isinstance(grabbed_objects, list):
        # t = dict()
        s = ""
        if len(grabbed_objects) == 0:
            s += "It's holding nothing. "
        else:
            s += f"It's holding <{grabbed_objects[0]['class_name']}> ({grabbed_objects[0]['id']}). "
            # t[grabbed_objects[0]['id']] =  grabbed_objects[0]['class_name']

            if len(grabbed_objects) == 2:
                s = s[:-2] + f" and <{grabbed_objects[1]['class_name']}> ({grabbed_objects[1]['id']}). "
                # t[grabbed_objects[1]['id']] =  grabbed_objects[1]['class_name']
        return s # , t
    elif isinstance(grabbed_objects, str):
        return grabbed_objects
    else:
        print("error in construct_holding_description")
        print("grabbed_objects",grabbed_objects)
        raise Exception("error in construct_holding_description")
