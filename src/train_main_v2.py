import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pickle
import time

from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt  # 导入 matplotlib
# 修改 Encoder 结构
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加归一化
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 定义强化学习模型
class ReinforcementLearningModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ReinforcementLearningModel, self).__init__()
        # 定义共享编码器
        # 定义共享编码器
        # 定义输出层
        self.frame_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.agent_id_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.state_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.task_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.action_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        # self.fc_fusion = nn.Sequential(
        #     nn.Linear(5 * hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)
        # )
    
    def forward(self, frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding):
        frame_encoded = self.frame_encoder(frame_embedding)
        agent_id_encoded = self.agent_id_encoder(agent_id_embedding)
        state_encoded = self.state_encoder(state_embedding)
        task_encoded = self.task_encoder(task_embedding)
        action_encoded = self.action_encoder(action_embedding)

        # 改用拼接+全连接融合
        # fused = torch.cat([
        #     frame_encoded,
        #     agent_id_encoded,
        #     state_encoded,
        #     task_encoded,
        #     action_encoded
        # ], dim=1)
        
        # current_q_value = self.fc_fusion(fused)
        # return current_q_value

        current_q_value = (frame_encoded * agent_id_encoded * state_encoded * action_encoded * task_encoded)
        current_q_value = current_q_value.squeeze(1)
        current_q_value = torch.mean(current_q_value, dim=1, keepdim=True)
        current_q_value = current_q_value * self.output_scale + self.output_bias

        # current_q_value = self.output_layer(current_q_value)
        # current_q_value = current_q_value * self.scale_factor  # 缩放到 [-100, 100]
        return current_q_value
    
    
# 加载编码后的数据
with open('/media/airs/BIN/graduation_design_env/encoded_data_v4.pkl', 'rb') as f:
    encoded_data = pickle.load(f)






# 预处理数据并加载到显存
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 将数据转换为张量并预加载到显存
def preprocess_data(data):
    processed_data = []
    for item in data.values():
        processed_data.extend(item)
    return processed_data

processed_data = preprocess_data(encoded_data)

# 将所有数据转换为张量并预加载到显存
frame_embeddings_t = []
agent_id_embeddings_t = []
state_embeddings_t = []
task_embeddings_t = []
action_embeddings_t = []
rewards_t = []

frame_embeddings_t_1 = []
agent_id_embeddings_t_1 = []
state_embeddings_t_1 = []
task_embeddings_t_1 = []
action_embeddings_t_1 = []
rewards_t_1 = []


for i in range(1, len(processed_data)):
    item_t_1 = processed_data[i - 1]
    frame_embeddings_t_1.append(torch.tensor(item_t_1['frame_embedding'], dtype=torch.float32))
    agent_id_embeddings_t_1.append(torch.tensor(item_t_1['agent_id_embedding'], dtype=torch.float32))
    state_embeddings_t_1.append(torch.tensor(item_t_1['state_embedding'], dtype=torch.float32))
    task_embeddings_t_1.append(torch.tensor(item_t_1['task_embedding'], dtype=torch.float32))
    action_embeddings_t_1.append(torch.tensor(item_t_1['action_embedding'], dtype=torch.float32))
    rewards_t_1.append(torch.tensor(item_t_1['reward'], dtype=torch.float32))
    
    item_t = processed_data[i]
    frame_embeddings_t.append(torch.tensor(item_t['frame_embedding'], dtype=torch.float32))
    agent_id_embeddings_t.append(torch.tensor(item_t['agent_id_embedding'], dtype=torch.float32))
    state_embeddings_t.append(torch.tensor(item_t['state_embedding'], dtype=torch.float32))
    task_embeddings_t.append(torch.tensor(item_t['task_embedding'], dtype=torch.float32))
    action_embeddings_t.append(torch.tensor(item_t['action_embedding'], dtype=torch.float32))
    rewards_t.append(torch.tensor(item_t['reward'], dtype=torch.float32))

frame_embeddings_t_1 = torch.stack(frame_embeddings_t_1).to(device)
agent_id_embeddings_t_1 = torch.stack(agent_id_embeddings_t_1).to(device)
state_embeddings_t_1 = torch.stack(state_embeddings_t_1).to(device)
task_embeddings_t_1 = torch.stack(task_embeddings_t_1).to(device)
action_embeddings_t_1 = torch.stack(action_embeddings_t_1).to(device)
rewards_t_1 = torch.stack(rewards_t_1).to(device)

# 堆叠张量
frame_embeddings_t = torch.stack(frame_embeddings_t).to(device)
agent_id_embeddings_t = torch.stack(agent_id_embeddings_t).to(device)
state_embeddings_t = torch.stack(state_embeddings_t).to(device)
task_embeddings_t = torch.stack(task_embeddings_t).to(device)
action_embeddings_t = torch.stack(action_embeddings_t).to(device)
rewards_t = torch.stack(rewards_t).to(device)



# 创建数据集
dataset = TensorDataset(
    frame_embeddings_t_1,
    agent_id_embeddings_t_1,
    state_embeddings_t_1,
    task_embeddings_t_1,
    action_embeddings_t_1,
    rewards_t_1,
    frame_embeddings_t,
    agent_id_embeddings_t,
    state_embeddings_t,
    task_embeddings_t,
    action_embeddings_t,
    rewards_t,
)

# 使用 PyTorch 的 DataLoader
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 设置超参数
embedding_dim = 768
hidden_dim = 128
learning_rate = 3e-4
num_epochs = 5000 * 4
TAU = 0.05  # 软更新系数


# 初始化模型、损失函数和优化器
online_model = ReinforcementLearningModel(embedding_dim, hidden_dim).to(device)
target_model = ReinforcementLearningModel(embedding_dim, hidden_dim).to(device)
target_model.load_state_dict(online_model.state_dict())

criterion = nn.SmoothL1Loss(beta=5.0)  # 对异常值更鲁棒
# 改用 Adam 优化器并添加权重衰减
optimizer = optim.AdamW(online_model.parameters(), 
                       lr=3e-4, 
                       weight_decay=1e-4)
loss_history = []

# 替代原有的硬更新
def soft_update(target, source):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(TAU * s.data + (1 - TAU) * t.data)


# 训练模型
for epoch in range(num_epochs):
    online_model.train()
    target_model.eval()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    cnt = 0

    for batch in progress_bar:
        frame_embedding_t_1 , agent_id_embedding_t_1, state_embedding_t_1, task_embedding_t_1, action_embedding_t_1, reward_t_1, frame_embedding_t, agent_id_embedding_t, state_embedding_t, task_embedding_t, action_embedding_t, reward_t = batch
        optimizer.zero_grad()
        
        # 计算当前Q值（使用在线网络）
        current_q = online_model(
            frame_embedding_t_1,
            agent_id_embedding_t_1,
            state_embedding_t_1,
            task_embedding_t_1,
            action_embedding_t_1
        )
        
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q = target_model(
                frame_embedding_t,
                agent_id_embedding_t,
                state_embedding_t,
                task_embedding_t,
                action_embedding_t
            )
            next_q = next_q.squeeze()
            target_q = reward_t_1 + 0.98 * next_q
            if cnt == 0:
                print("| Model Output | Target Values |")
                print("|--------------|---------------|")
                print(f"| {current_q.mean():.1f}±{current_q.std():.1f} | {target_q.mean():.1f}±{target_q.std():.1f} |")
                cnt += 1
                plt.figure(figsize=(12,6))
                plt.hist(current_q.cpu().numpy(), bins=50, alpha=0.5, label='Model Output')
                plt.hist(target_q.cpu().numpy(), bins=50, alpha=0.5, label='Target Values')
                plt.legend()
                plt.savefig('distribution_alignment.png')
        # 计算损失并优化
        loss = criterion(current_q, target_q)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_model.parameters(), 5.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # 保存每个 epoch 的平均损失
    avg_epoch_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_epoch_loss)

    # 定期更新目标网络
    if epoch % 500 == 0:
        # 训练循环中持续更新
        target_model.load_state_dict(online_model.state_dict())
        model_name = f"V2_{time.time()}_epoch_{epoch}.pth"
        # 保存模型
        torch.save(online_model.state_dict(), model_name)
        print("模型已保存")

        # soft_update(target_model, online_model)    
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')





# 绘制损失曲线
# plt.figure(figsize=(18, 6))
# plt.plot(range(1, num_epochs + 1), loss_history)
# plt.title('Training Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.grid(True)
# plt.savefig('training_loss.png')  # 保存损失曲线图
# plt.show()