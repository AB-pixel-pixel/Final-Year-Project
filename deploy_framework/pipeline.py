#  "/home/airs/miniforge3/envs/ai2thtor_bin/lib/python3.9/site-packages",
import importlib
from typing import Any
from framework_structure import *
from framework_common_imports import *
from framework_dispatching_task import *

from framework_step2action import step2action_by_similarity_v4,random_select,select_based_on_word_frequency,step2action_by_similarity_v5,select_based_on_word_frequency_v1
from framework_refine_step import *
from process_text import remove_labels , get_name_to_id,  get_id_to_name



class Pipeline():
    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self.memory = Memory()
        self.FLAG_INIT : bool = True
        self.agents_: list[Agent] = []
        for i in range(ROBOT_NUMBER):
            agent = Agent(agent_id= i)
            self.agents_.append(agent)
        # 用於存放一些公共的屬性,例如,human_instruction這些,這樣就不用維護每個agent的相關公共屬性是否一致
        # 同時agent中的公共屬性是不應該拿來使用的
        self.public_pipeline_state : AGENT_STATE = get_initial_agent_state()
        # self.forzen_public_pipeline_state : AGENT_STATE = None
        current_environment = os.getenv('experiment_name', 'tdw')  # 默认值为 'tdw'
        wrapper_module = ""
        if current_environment == 'tdw':
            wrapper_module = 'interface_tdw_wrapper'
        elif current_environment == 'cwah':
            wrapper_module = 'interface_cwah_wrapper'
        else:
            raise ValueError(f"Unknown environment: {current_environment}")
        self.module = importlib.import_module(wrapper_module)




    def _setup(self,msg: BATCH_PROMPT_Constituent_Elements):
        self.public_pipeline_state = self.module.get_initial_public_pipeline()
        self.public_pipeline_state['overall_observation'] = self.module.get_agents_description(msg) 
        self.public_pipeline_state['action_history'] = []
        log_system.LOG(self.public_pipeline_state['overall_observation'])
        
        for agent in self.agents_:
            agent.state = copy.deepcopy(self.public_pipeline_state) # 之前缺乏了深度拷贝,导致后面改变的状态会改变前面的状态
        set_all_agents_stage(self.agents_,target_stage=Stage.initial_dispatching_task)
        print("------------------------------setup pipeline------------------------------")


    def receive(self,msg: BATCH_PROMPT_Constituent_Elements) -> Optional[Action_Options]:
        # log_system.PL("------------------------------receive pipeline------------------------------\nBATCH_PROMPT_Constituent_Elements\n",msg)
        if self.FLAG_INIT:
            self._setup(msg)
            print(self.FLAG_INIT)
        self.update(msg)

        log_system.PL("------step_num:",self.public_pipeline_state["step_num"],"------")

        if self.FLAG_INIT:
            self.FLAG_INIT = False
            ans = self.run(state_=1,msg=msg)
        else:
            # 明確不需要重新規劃的 agent
            self.filter_agents_requiring_next_action(msg=msg)
            ans = self.run(state_=5,msg=msg) # 从5号状态进入
        if ans:
            return  ans
        else:
            return None

    def update(self,msg: BATCH_PROMPT_Constituent_Elements):
        # TODO 简化这里的代码
        # just update overall_observation
        self.public_pipeline_state['current_task'] = msg.batch_prompt_constituent_elements[0].current_task
        self.public_pipeline_state["step_num"] = msg.batch_prompt_constituent_elements[0].step_num

        # update agent
        batch_prompt = msg.batch_prompt_constituent_elements
        for prompt_elements in batch_prompt:
            agent = self.agents_[get_name_to_id(prompt_elements.agent_name)]
            update_state(agent,prompt_elements)
            # for agent in self.agents_:
            #     if get_id_to_name(agent.agent_id) in prompt_elements.agent_name:
                    # 在这一步更新state and observation, available plan

        # update memory
        for prompt_elements in batch_prompt:
            self.memory.update_room_cost(prompt_elements)
            self.memory.update_observation(prompt_elements)
            # self.memory.update_holding(prompt_elements)
            log_system.LOG(self.memory.get_observation_text())
        
        # update task relevant observation
        self.public_pipeline_state['task_relevant_obsevation'] = self.memory.get_task_relevant_info(self.public_pipeline_state['current_task'])

        for prompt_elements in batch_prompt:
            agent_id = get_name_to_id(prompt_elements.agent_name)
            agent = self.agents_[agent_id]
            relevant_info = self.memory.get_available_actions_relevant_info(agent_id,prompt_elements.available_plans)
            # available_plans = prompt_elements.available_plans.split("\n")
            # new_available_plans = []
            agent.state['available_plans_with_info'] = relevant_info + prompt_elements.available_plans
            

        # Update overall observation
        overall_observation = ""
        for agent in self.agents_:
            overall_observation += agent.get_description()
            # agent.state['observation'] = agent.get_description()
            # log_system.PRINT(f"{agent.agent_id} agent.state['observation']",agent.state['observation'])
            
        self.public_pipeline_state['overall_observation'] = overall_observation
        
        # log_system.PL(f"agent.state['available_plans']: {agent.state['available_plans']}")
    
    # FLAG_DECOMPOSED_ALL_TASKS : bool = False
    FLAG_NEED_NEW_COT :bool = False
    last_ISO_id : int = 0
    FLAG_NEED_REALLOCATE : bool = False

            
    
    def run(self,state_ : int ,msg :BATCH_PROMPT_Constituent_Elements ) -> Action_Options:
        """
        一个简单的状态机示例,输入msg,输出action。

        状态定义：
        1: 任务分配(初始化)
        2: 任务分配(單個agent或多個agent)
        3: Cot生成(初始化)
        4: Cot生成(單個agent或多個agent)
        5: 選取Cot下一步
        6: Cot动作生成

        """
        state = StateMachineState(state_)
        
        for agent in self.agents_:
            agent.initial_planning_counter()

        ans = dict() # 對特定的agent進行動作生成
        while(True):
            for agent in self.agents_:
                log_system.PL("agent:",agent.agent_id,"stage:",agent.stage)

            if state == StateMachineState.dispatching_i: # 初始化任务分配
                log_system.PL("-----------state 1 : 任务分配(初始化) -------------------")
                self.public_pipeline_state['state'] = self.agents_[0].state['state']
                
                self.public_pipeline_state["task_allocation_counter"] = 1
                self.public_pipeline_state['human_instruction'] = self.public_pipeline_state['current_task']
                task_decomposed_result = new_dispatching_task(self.public_pipeline_state,self.agents_)
                

                for id_task_pair in task_decomposed_result.robot_id_task_pairs:
                    robot_id = id_task_pair.robot_id
                    task = id_task_pair.task_content # action chain class
                    agent = self.agents_[robot_id]
                    
                    agent.set_action_tree(task)
                    agent.stage = Stage.get_action
                    agent.state['task_content'] = action_chain_to_text(task)
                    self.public_pipeline_state['task_allocation_result'][robot_id] = task

                    log_system.PL(f"Agent {robot_id} 已經分配得到 {task}")

                    log_system.add_log(public_pipeline_state = self.public_pipeline_state,agent = agent,supplementary_explanation="任务分配")
                # log_system.PL("self.public_pipeline_state", self.public_pipeline_state)

                state = StateMachineState.get_action
                set_all_agents_stage(self.agents_,Stage.get_action)


            elif state == StateMachineState.dispatching:


                log_system.PL("-----------state 2 : 任务分配 -------------------")
                
                self.public_pipeline_state["task_allocation_counter"] += 1

                task_decomposed_result = new_dispatching_task(self.public_pipeline_state,self.agents_)
                

                self.public_pipeline_state['human_instruction'] = self.public_pipeline_state['current_task']  #task_decomposed_result.the_rest_tasks
                self.allocate_task(msg,task_decomposed_result)

              

                state = StateMachineState.get_action
                log_system.PRINT(f"next state is {state}")


            elif state == StateMachineState.get_action:
                
                log_system.PL("-----------state 3 Get Action-------------------")
                _next_state = StateMachineState.attempt_execution


                get_action_agents = [agent for agent in self.agents_  if agent.stage == Stage.get_action]
                for agent in get_action_agents:
                    log_system.LOG(agent.display_action_chain_with_block_info())
                    agent_id = agent.agent_id
                    step = agent.action_tree.get_next_action()
                    log_system.PRINT("current action is ", step)
                    # step 有可能为 None
                    if step:
                        log_system.PL(f"-------------------Agent {agent_id} Next STEP-------------------\n",step)
                        agent.stage = Stage.check_action # 前往执行动作

                    else:
                        agent.stage = Stage.dispatching_task
                        _next_state = state_min(StateMachineState.dispatching, _next_state) # 如果出現失敗的就跳轉到2,优先级最高
                        
                        log_system.PRINT(f"-------------------Agent {agent_id} 分配任务去了-------------------\n")
                        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="执行完成所有步骤，重新分配任务")

                    log_system.PL(f"-------------------{agent_id} 的状态为: {agent.stage}")

                state = _next_state
                log_system.PRINT(f"next state is {_next_state}")

            elif state == StateMachineState.attempt_execution:
                # 动作生成,经过前面的步骤,所有Agent都已经获取到当前的动作了
                log_system.PL("-----------state 6: attempt to execute -------------------")

                _next_state = StateMachineState.finish_state
                
                for agent in  self.agents_:
                    if not agent.state['need_replan']:
                        continue
                    log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation=f"Agent{agent.agent_id} attempt to execute")

                    if agent.stage != Stage.check_action:
                        continue

                    log_system.LOG(f"Agent id: {agent.agent_id} 生成动作")

                    target_step = agent.action_tree.get_next_action()

                    
                    available_plans = agent.state['available_plans']

                    log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation=f"动作匹配,target action : {target_step}\n可选动作为:{available_plans}")

                    option, similarity, need_replan, cleared_option = step2action_by_similarity_v5(target_step,available_plans)
                    
                    if need_replan:
                        log_system.PL(f"----------- Agent{agent.agent_id} Replan-------------------")
                        # Handle 'Replan' command.

                        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Start Replan")

                        agent.set_disposable_planning_reason("The current agent requests to plan a few new actions based on the current observation.")
                        self.create_and_merge_action_chain(agent=agent)

                        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Finish Replan")

                        agent.stage = Stage.get_action
                        _next_state = state_min(_next_state, StateMachineState.get_action)

                    elif option:
                        # 成功匹配
                        # check conflict
                        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation=f"Matching to action: {option}")

                        
                        # check explore and exploitation
                        if agent.action_tree.evaluate_explore_exploitation():
                            log_system.add_log(public_pipeline_state = self.public_pipeline_state,agent = agent,supplementary_explanation="动作树评估,决定是否需要重新规划")

                            # 动作树评估,决定是否需要重新规划
                            temp_flag, temp_ans = self.check_over_replan(agent)
                            if temp_flag:
                                ans.update(temp_ans)
                                continue
                            else:
                                agent.set_disposable_planning_reason("The agent's current action sequence is too focused on exploration and lacks exploitation, with many exploratory actions that don't involve grasping objects. Please adjust the subsequent actions to prioritize grasping relevant objects. Instead of exploring the room one by one, explore or go to the room and then grasp one or two traget objects, transport them to the target location. If the items in hand exceed carrying capacity, transport them to the target location.")
                                self.refine_agent(agent)

                                log_system.add_log(public_pipeline_state = self.public_pipeline_state,agent = agent,supplementary_explanation="Explore too much ! Replan")

                                agent.stage = Stage.get_action
                                _next_state = state_min(_next_state,StateMachineState.get_action)
                                continue

                        log_system.PL("option: \n",option)
                        if "transport" in option.lower() or 'bedroom' in option.lower(): # transport action don't involve with conflict checking
                            log_system.PL("Skip conflict checking because of transport action ")
                            log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Skip conflict checking because of transport action ")
                        else:
                            temp_ans , temp_state = self.check_conflict(agent,cleared_option,option)
                            if temp_ans is None and temp_state is None:
                                pass
                            elif temp_ans is None:
                                # not over replan , but need to adjust the action
                                _next_state = state_min(_next_state,temp_state)
                                continue
                            elif temp_state is None:
                                # over replan
                                ans.update(temp_ans)
                                continue


                        # execute it
                        agent.stage = Stage.execute_action
                        agent.next()
                        agent.action = Agent_Decision(agent_id=agent.agent_id,decision=option,reason="")
                        ans[get_id_to_name(agent.agent_id)] = agent.action.decision
                        agent.state["action_history"].append(option)
                        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Execute the action which pass all the test.")
                     
                
                    else:
                        temp_flag, temp_ans = self.check_over_replan(agent)
                        if temp_flag:
                            # if over replan , return similarity ans
                            ans.update(temp_ans)
                        else:
                            log_system.PL(f"----------- 动作匹配失败,Agent{agent.agent_id} 重新规划新动作 -------------------")
                            if "transport" in target_step:
                                agent.set_disposable_planning_reason("Action is not suitable for the available action. If the original action including \"transport something to bed \" is unavailable in action space, you can replace it with a action sequence of \"go to bedroom\", \"transport objects I'm holding to the bed.\" ")
                            elif "grasp" in target_step:
                                agent.set_disposable_planning_reason("Replace the action chain into some similary actions from action space. The origin action chain have typo error.")

                            self.refine_agent(agent)

                            log_system.add_log(public_pipeline_state = self.public_pipeline_state,agent = agent,supplementary_explanation="Fail to match a good action, just replan it.")

                            agent.stage = Stage.get_action
                            _next_state = state_min(_next_state,StateMachineState.get_action)

                if _next_state == StateMachineState.finish_state:
                    break
                else:
                    state = _next_state
                    log_system.PL(f"next state is {_next_state}")


            else:
                log_system.PL(f"-------------------出現錯誤的state:{state}-------------------\n")
        
        # 画图
        # save_graph(ROOT_NODE=temp_root_node,prefix_file="/home/airs/bin/ai2thor_env/cb/planning_logs",current_nodes=[],frame_number=self.public_pipeline_state["step_num"])
        
        return Action_Options(batch_actions=ans) # 当无法奏效时,将会返回{}
    
    
    def allocate_task(self,msg,task_decomposed_result):
        for prompt_elements in  msg.batch_prompt_constituent_elements:
            if prompt_elements.need_replan:
                target_agent_name = prompt_elements.agent_name
                for id_task_pair in task_decomposed_result.robot_id_task_pairs:
                    robot_id = id_task_pair.robot_id
                    task = id_task_pair.task_content
                    robot_name = get_id_to_name(robot_id)
                    if robot_name == target_agent_name:
                        self.public_pipeline_state['task_allocation_result'][robot_id] = task
                        agent = self.agents_[robot_id]
                        agent.state['task_content'] = task
                        agent.state["current_task"] = self.public_pipeline_state['current_task']

                        agent.action_tree.add_new_chain(task)
                        agent.stage = Stage.get_action
                        log_system.PL(f"Agent {robot_id} 已經分配得到 {task}")
                        log_system.add_log(public_pipeline_state = self.public_pipeline_state,agent = agent,supplementary_explanation="任务分配")


    def refine_action_step(self,agent: Agent) -> Action:
        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Calling refine_action_step function.") 

        """ 方法退化至仅对下一步进行单步修改以满足匹配选项的需求 """
        agent.planning_counter += 1
        prompt, new_action_step = action_step_mutation(agent)
        agent.action_tree.refine_node(single_step=new_action_step)
        # TODO put prompt into log
        return new_action_step
    


    def skip_node(self, agent : Agent):
        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Calling skip_node function.") 

        agent.action_tree.skip_node()
        agent.get_disposable_planning_reason()

        
    def create_and_merge_action_chain(self,agent: Agent):
        """ handle 'replan' command """

        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Calling create_and_merge_action_chain function.") 

        log_system.PL("----------- create_action_chain -------------------")

        available_plans = agent.state['available_plans']
        prompt , new_action_chain = create_action_chain(agent,available_plans)
        
        if new_action_chain is None or "none" in [step.step.lower() for step in new_action_chain.chain]:
            # 如果空白就跳过那些特殊的结点
            agent.action_tree.skip_node()
            log_system.PL("-------------- skip node -----------------")
        else:
            log_system.PL(f"Before:\n{agent.action_tree.get_refine_action_info(100)}")

            log_system.PL(new_action_chain)

            agent.action_tree.merge_action_chain(new_action_chain) # Handle replan
            
            log_system.PL(f"After:\n{agent.action_tree.get_refine_action_info(100)}")
        agent.state['replan_reason'] = ""
        agent.action_tree.display_overall_action_tree()


    def solver_conflict(self, agent : Agent, conflict_agent : Agent, conflict_message : str):
        """ solve conflict """
        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Calling solver_conflict function.") 

        prompt, ans = solve_confilct(agent,conflict_agent,conflict_message)
        # method = matching_refine_method_v2(ans.advice_agent0)
        
        agent.action_tree.refine_action_chain(ans.action_chain0)
        agent.state['replan_reason'] = ""
        
        agent.action_tree.refine_action_chain(ans.action_chain1)
        agent.state['replan_reason'] = ""
    
    
    def filter_agents_requiring_next_action(self,msg):
        """将需要重新规划的Agent提取出来"""
        agent_id_filter = [get_name_to_id(_msg.agent_name) for _msg in msg.batch_prompt_constituent_elements if _msg.need_replan]
        for agent in self.agents_:
            if agent.agent_id in agent_id_filter:
                agent.stage = Stage.get_action


    def close(self): 
        print("------------------------------close pipeline------------------------------")
        planning_data = log_system.save_log()
        log_system.close()
        self.FLAG_INIT = True
        return planning_data



    def refine_agent(self, agent: Agent):
        """
        return skip_action_step or 'refine_action_chain' or 'refine_action_step'
        """
        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Calling refine_agent function.") 

        log_system.PL(f"----------- refine_agent -------------------")

        agent.planning_counter += 1

        prompt, refine_result = refining(agent)
        agent.action_tree.refine_action_chain(refine_result.action_chain)


    def check_over_replan(self,agent : Agent)-> Tuple[bool,dict]:
        """ use the function before refine any action 
            True for over replan , ans will be dict
            False for not over replan, ans will be None
        """
        # if agent.is_reach_limit():
        #     # 超出规划步数限制
        #     log_system.PL(f"-------------------Agent {agent.agent_id} arrive 推理次数限制，退化为仅修改当前结点 -------------------\n")
        #     log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Arrive推理次数限制，退化为仅修改当前结点")

        #     # agent.set_disposable_planning_reason("The action about to be executed does not match any of the available actions.")
        #     self.refine_action_step(agent)
        #     log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="完成当前未匹配动作的修改")

        #     agent.stage = Stage.get_action
        #     _next_state = state_min(_next_state, StateMachineState.get_action)
        #     return (True, _next_state)
        if agent.is_exceeding_limit():
            # 超出执行步数限制, 随机选择
            log_system.PL(f"-------------------Agent {agent.agent_id} 超出规划限度, 基于词频的相似度对目标选项进行选择-------------------\n")
            log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="agent is over replan ")
            target_step = agent.action_tree.get_next_action()
            available_plans = agent.state['available_plans']
            option = select_based_on_word_frequency_v1(target_step, available_plans)
            log_system.PL("target_step: ", target_step,"\noption: ",option)
            agent.stage = Stage.execute_action
            agent.action_tree.replace_node(option)
            
            agent.next()
            agent.action = Agent_Decision(agent_id=agent.agent_id,decision=option,reason=str(0))
            ans = {get_id_to_name(agent.agent_id):agent.action.decision}
            agent.state["action_history"].append(option)
            log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation=f"Due to over replan, select {option} based on word frequency")
            # 跳过后续匹配
            return (True, ans)
        log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation=f"Over replan don't exist")
        return (False, {})


    def check_conflict(self,agent : Agent, cleared_option : str,option : str):
        # 检索
        _state_flag = 1
        _conflict_agent_id = []
        conflict_messages : list[str] = []
        for target_agent in self.agents_:
            if agent.agent_id != target_agent.agent_id:
                temp_flag, conflict_messages = target_agent.action_tree.retrieve_nodes_by_content(cleared_option) #  只检索当前相近的action step
                _state_flag = node_state_max(_state_flag,temp_flag)
                _conflict_agent_id.append(target_agent.agent_id)


        # # 数字大小表示覆盖顺序（优先级）
        # head_node = 0
        # # 动作节点未执行
        # not_executed = 1
        # # 动作节点无效
        # invalid = 2
        # # 动作结点已经执行过（ 给 replan 专用 )
        # executed = 3
        # # 动作节点有效
        # valid = 4
        if _state_flag == 4:
            # # TODO provide the conflict node
            handle_conflict_flag = agent.check_exemption_list(option)
            if handle_conflict_flag:
                temp_flag, temp_ans = self.check_over_replan(agent)
                if temp_flag:
                    # if over replan , return similarity ans
                    return temp_ans,None
                else:
                    log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="Conflict detected, replan it.")
                    # if not over replan , replan it
                    conflict_agent_id = _conflict_agent_id[0]
                    conflict_agent = self.agents_[conflict_agent_id]
                    self.solver_conflict(agent,conflict_agent,conflict_messages[0]) # TODO maybe need take the conflict message seriously
                    if conflict_agent.stage != Stage.execute_action:     
                        conflict_agent.stage = Stage.get_action       
                    agent.stage = Stage.get_action
                    # _next_state = state_min(_next_state,StateMachineState.get_action)
                    return None,StateMachineState.get_action
        return None,None