from framework_structure import *
from framework_common_imports import *
from process_text import remove_labels , get_name_to_id,  get_id_to_name

from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from interface_tdw_wrapper import get_initial_public_pipeline, get_agents_description

from models.basic_dqn import ReinforcementLearningModel

class Pipeline():
    def __init__(self) -> None:
        # 全局模型和分词器
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_path = "/media/airs/BIN/LargeModels/embedding_models"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # 加载嵌入模型
        self.embedding_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.embedding_model.eval()
        # 加载强化学习模型
        self.rl_model = ReinforcementLearningModel(768, 128).to(self.device)
        self.rl_model.load_state_dict(torch.load("/media/airs/BIN/graduation_design_env/saved_models/trainning_v1/V2_1743867171.4562988_epoch_3500.pth", map_location=self.device,weights_only=True))  # 替换为你的模型路径
        self.rl_model.eval()

        self.FLAG_INIT : bool = True
        self.agents_: list[Agent] = []
        for i in range(ROBOT_NUMBER):
            agent = Agent(agent_id= i)
            self.agents_.append(agent)
        # 用於存放一些公共的屬性,例如,human_instruction這些,這樣就不用維護每個agent的相關公共屬性是否一致
        # 同時agent中的公共屬性是不應該拿來使用的
        self.public_pipeline_state : AGENT_STATE = get_initial_agent_state()
        self.mode = 1
        self.epsilon = 0.1


    def _setup(self,msg: BATCH_PROMPT_Constituent_Elements):
        self.public_pipeline_state = get_initial_public_pipeline()
        self.public_pipeline_state['overall_observation'] = get_agents_description(msg) 
        self.public_pipeline_state['action_history'] = []
        log_system.LOG(self.public_pipeline_state['overall_observation'])
        
        for agent in self.agents_:
            agent.state = copy.deepcopy(self.public_pipeline_state) # 之前缺乏了深度拷贝,导致后面改变的状态会改变前面的状态
        set_all_agents_stage(self.agents_,target_stage=Stage.initial_dispatching_task)
        print("------------------------------setup pipeline------------------------------")


    def receive(self,msg: BATCH_PROMPT_Constituent_Elements) -> Optional[Action_Options]:
        if self.FLAG_INIT:
            self._setup(msg)
            print(self.FLAG_INIT)
        self.update(msg)
        log_system.PL("------step_num:",self.public_pipeline_state["step_num"],"------")
        return self.wrap_select_action(msg)



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
    

    def wrap_select_action(self,request :  BATCH_PROMPT_Constituent_Elements):
        # 判断agent是否超出执行步数限制
        log_system.PL("------step_num:",self.public_pipeline_state["step_num"],"------")

        agents  = [agent for agent in self.agents_ if agent.state.get('need_replan',True)]
        ans = dict()
        msgs = request.batch_prompt_constituent_elements
        if len(msgs) == 1:
            option = self.select_action(msgs[0])
            ans[get_id_to_name(agent.agent_id)] = option
            log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="") 
        else:
            for agent in agents:
                option = self.select_action(msgs[agent.agent_id])
                ans[get_id_to_name(agent.agent_id)] = option
                log_system.add_log(public_pipeline_state=self.public_pipeline_state,agent=agent,supplementary_explanation="") 

        return Action_Options(batch_actions=ans) # 当无法奏效时,将会返回{}


    def select_action(self,request: PROMPT_Constituent_Elements):
        
        # 使用 re.sub 去掉匹配到的部分
        pattern = r'[A-Z]\.\s'
        cleaned_text = re.sub(pattern, '', request.available_plans)
        options = [option.lower().replace("<","").replace(">","").replace("_"," ") for option in cleaned_text.split("\n") if option.strip() ]
        print("options:\n", options)
        # 生成嵌入
        with torch.no_grad():
            frame_embed = self.get_embedding(str(request.step_num))
            agent_embed = self.get_embedding(str(get_name_to_id(request.agent_name)))
            state_embed = self.get_embedding(request.progress_desc)
            task_embed = self.get_embedding(request.current_task)
            action_embeds = torch.cat([self.get_embedding(option) for option in options] )
        
        # 扩展固定嵌入维度
        batch_size = len(options)
        frame_embed = frame_embed.expand(batch_size, -1)
        agent_embed = agent_embed.expand(batch_size, -1)
        state_embed = state_embed.expand(batch_size, -1)
        task_embed = task_embed.expand(batch_size, -1)
        
        # 计算Q值
        q_values = self.rl_model(frame_embed, agent_embed, state_embed, task_embed, action_embeds)
        q_values = q_values.squeeze().cpu().detach().numpy().tolist()
        print("q_values\n",q_values)
        
        selected_idx = 0
        # 选择动作
        if self.mode == 1:
            selected_idx = np.argmax(q_values)
        elif self.mode == 2:
            if np.random.rand() < self.epsilon:
                selected_idx = np.random.randint(len(q_values))
            else:
                selected_idx = np.argmax(q_values)
        else:
            raise ValueError("Invalid mode")
        return request.available_plans.split('\n')[selected_idx]
    

    
    def get_embedding(self,text: str):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=4000, truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
        return F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
    
    def close(self):
        print("------------------------------close pipeline------------------------------")
        planning_data = log_system.save_log()
        log_system.close()
        self.FLAG_INIT = True
        return planning_data