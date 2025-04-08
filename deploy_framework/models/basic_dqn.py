import torch
import torch.nn.functional as F


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