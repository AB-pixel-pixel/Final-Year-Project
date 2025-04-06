from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np
import pickle
import re

from communication_protocol import PROMPT_Constituent_Elements,BATCH_PROMPT_Constituent_Elements


# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定义强化学习模型
class ReinforcementLearningModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ReinforcementLearningModel, self).__init__()
        self.frame_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.agent_id_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.state_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.task_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.action_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.candidate_action_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
    
    def forward(self, frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding, candidate_actions_embedding):
        frame_encoded = self.frame_encoder(frame_embedding)
        agent_id_encoded = self.agent_id_encoder(agent_id_embedding)
        state_encoded = self.state_encoder(state_embedding)
        task_encoded = self.task_encoder(task_embedding)
        action_encoded = self.action_encoder(action_embedding)
        
        candidate_actions_encoded = [self.candidate_action_encoder(ca) for ca in candidate_actions_embedding]
        
        q_values = []
        for ca_encoded in candidate_actions_encoded:
            q_value = (frame_encoded * agent_id_encoded * state_encoded * task_encoded * action_encoded * ca_encoded)
            q_values.append(q_value).sum()
        
        return torch.stack(q_values)

# 加载预训练模型和分词器
model_name_or_path = 'iic/gte_sentence-embedding_multilingual-base'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
embedding_model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
dimension = 768

# 定义函数，将文本转换为embedding
def get_embedding(texts):
    batch_dict = tokenizer(texts, max_length=8192, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        outputs = embedding_model(**batch_dict)
    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.numpy()

# 清理 available_plans 的函数
def remove_labels(text):
    pattern = r'[A-Z]\.\s'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

# 初始化FastAPI应用
app = FastAPI()

# 加载强化学习模型
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = ReinforcementLearningModel(embedding_dim=768, hidden_dim=256).to(device)
model.load_state_dict(torch.load('reinforcement_learning_model.pth', map_location=device))
model.eval()

# 推理函数
def inference(prompt_element):
    # 对输入信息进行编码
    frame_embedding = get_embedding([prompt_element.agent_name])
    agent_id_embedding = get_embedding([prompt_element.current_room])
    state_embedding = get_embedding([prompt_element.progress_desc])
    task_embedding = get_embedding([prompt_element.current_task])
    
    # 清理 available_plans 并生成候选动作列表
    cleaned_available_plans = remove_labels(prompt_element.available_plans)
    options = [option for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option]
    
    # 如果没有有效的候选动作，直接返回
    if not options:
        return {"best_action": None, "q_values": None}
    
    # 对候选动作进行编码
    candidate_actions_embedding = get_embedding(options)
    
    # 转换为张量并移动到设备上
    frame_embedding = torch.tensor(frame_embedding, dtype=torch.float32).to(device)
    agent_id_embedding = torch.tensor(agent_id_embedding, dtype=torch.float32).to(device)
    state_embedding = torch.tensor(state_embedding, dtype=torch.float32).to(device)
    task_embedding = torch.tensor(task_embedding, dtype=torch.float32).to(device)
    action_embedding = torch.tensor(get_embedding([prompt_element.available_plans]), dtype=torch.float32).to(device)
    candidate_actions_embedding = [torch.tensor(ca, dtype=torch.float32).to(device) for ca in candidate_actions_embedding]
    
    # 模型推理
    with torch.no_grad():
        q_values = model(frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding, candidate_actions_embedding)
    
    # 获取最佳动作的索引和对应的动作描述
    best_action_idx = torch.argmax(q_values).item()
    best_action = options[best_action_idx]
    
    return {"best_action": best_action, "q_values": q_values.cpu().numpy().tolist()}

# API端点
@app.post("/infer/")
async def infer(batch_prompt: BATCH_PROMPT_Constituent_Elements):
    results = {}
    for prompt_element in batch_prompt.batch_prompt_constituent_elements:
        if prompt_element.need_replan:
            try:
                result = inference(prompt_element)
                if result["best_action"] is not None:
                    results[prompt_element.agent_name] = result["best_action"]
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    return {"results": results}