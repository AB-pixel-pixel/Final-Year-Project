import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset

# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()  # 使用 Sigmoid 激活函数
        self.scale = nn.Parameter(torch.ones(1))  # 初始化为1
        self.shift = nn.Parameter(torch.zeros(1))  # 初始化为0
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x) * self.scale + self.shift  # 应用缩放和偏移
        x = self.fc2(x)
        return x

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
    
    def forward(self, frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding):
        frame_encoded = self.frame_encoder(frame_embedding)
        agent_id_encoded = self.agent_id_encoder(agent_id_embedding)
        state_encoded = self.state_encoder(state_embedding)
        task_encoded = self.task_encoder(task_embedding)
        action_encoded = self.action_encoder(action_embedding)
        
        current_q_value = (frame_encoded * agent_id_encoded * state_encoded * action_encoded * task_encoded)
        current_q_value = current_q_value.squeeze(1)
        current_q_value = torch.mean(current_q_value, dim=1, keepdim=True)
        # current_q_value = self.output_layer(current_q_value)
        # current_q_value = current_q_value * self.scale_factor  # 缩放到 [-100, 100]
        return current_q_value
    
    
# 加载编码后的数据
with open('/media/airs/BIN/graduation_design_env/encoded_data_v3.pkl', 'rb') as f:
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
hidden_dim = 64
learning_rate = 3e-4
num_epochs = 50000

# 初始化模型、损失函数和优化器
online_model = ReinforcementLearningModel(embedding_dim, hidden_dim).to(device)
target_model = ReinforcementLearningModel(embedding_dim, hidden_dim).to(device)
target_model.load_state_dict(online_model.state_dict())

criterion = nn.MSELoss()
optimizer = optim.SGD(online_model.parameters(), lr=learning_rate)

loss_history = []

# 训练模型
for epoch in range(num_epochs):
    online_model.train()
    target_model.eval()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    
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
            target_q = reward_t_1 + 0.9 * next_q
        
        # 计算损失并优化
        loss = criterion(current_q, target_q)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    # 保存每个 epoch 的平均损失
    avg_epoch_loss = epoch_loss / len(dataloader)
    loss_history.append(avg_epoch_loss)

    # 定期更新目标网络
    if epoch % 200 == 0:
        target_model.load_state_dict(online_model.state_dict())
    
    # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(dataloader):.4f}')


model_name = "v1"
# 保存模型
torch.save(online_model.state_dict(), f'{model_name}.pth')
print("模型已保存")

import matplotlib.pyplot as plt  # 导入 matplotlib


# 绘制损失曲线
plt.figure(figsize=(18, 6))
plt.plot(range(1, num_epochs + 1), loss_history)
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('training_loss.png')  # 保存损失曲线图
plt.show()