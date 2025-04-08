from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 设备配置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义强化学习模型（与训练代码一致）
class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ReinforcementLearningModel(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.frame_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.agent_id_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.state_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.task_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.action_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.output_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.output_bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding):
        frame_encoded = self.frame_encoder(frame_embedding)
        agent_id_encoded = self.agent_id_encoder(agent_id_embedding)
        state_encoded = self.state_encoder(state_embedding)
        task_encoded = self.task_encoder(task_embedding)
        action_encoded = self.action_encoder(action_embedding)

        current_q_value = (frame_encoded * agent_id_encoded * state_encoded * action_encoded * task_encoded)
        current_q_value = torch.mean(current_q_value, dim=1, keepdim=True)
        return current_q_value * self.output_scale + self.output_bias

# 全局模型和分词器
model_path = "/media/airs/BIN/LargeModels/embedding_models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
# 加载嵌入模型
embedding_model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
embedding_model.eval()
# 加载强化学习模型
rl_model = ReinforcementLearningModel(768, 128).to(device)
rl_model.load_state_dict(torch.load("V2_1743867171.4562988_epoch_3500.pth", map_location=device,weights_only=True))  # 替换为你的模型路径
rl_model.eval()

class ActionRequest(BaseModel):
    frame: str
    agent_id: str
    state: str
    task: str
    candidate_actions: List[str]
    mode: int = 1
    temperature: Optional[float] = 1.0
    epsilon: Optional[float] = 0.1

class ActionResponse(BaseModel):
    selected_action: str
    q_values: List[float]
    details: dict

def get_embedding(text: str):
    global tokenizer, embedding_model, rl_model
    try:
        inputs = tokenizer(text, return_tensors="pt", max_length=4000, truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        return F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

@app.post("/select-action", response_model=ActionResponse)
async def select_action(request: ActionRequest):
    try:
        # 验证输入
        if not request.candidate_actions:
            raise HTTPException(status_code=400, detail="Candidate actions cannot be empty")
        
        # 生成嵌入
        with torch.no_grad():
            frame_embed = get_embedding(request.frame)
            agent_embed = get_embedding(request.agent_id)
            state_embed = get_embedding(request.state)
            task_embed = get_embedding(request.task)
            action_embeds = torch.cat([get_embedding(action) for action in request.candidate_actions])
        
        # 扩展固定嵌入维度
        batch_size = len(request.candidate_actions)
        frame_embed = frame_embed.expand(batch_size, -1)
        agent_embed = agent_embed.expand(batch_size, -1)
        state_embed = state_embed.expand(batch_size, -1)
        task_embed = task_embed.expand(batch_size, -1)
        
        # 计算Q值
        q_values = rl_model(frame_embed, agent_embed, state_embed, task_embed, action_embeds)
        q_values = q_values.squeeze().cpu().numpy().tolist()
        
        # 选择动作
        if request.mode == 1:
            selected_idx = np.argmax(q_values)
        elif request.mode == 2:
            if np.random.rand() < request.epsilon:
                selected_idx = np.random.randint(len(q_values))
            else:
                selected_idx = np.argmax(q_values)
        else:
            raise HTTPException(status_code=400, detail="Invalid selection mode")
        
        return ActionResponse(
            selected_action=request.candidate_actions[selected_idx],
            q_values=q_values,
            details={
                "candidate_actions": request.candidate_actions,
                "q_values": q_values
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9070)