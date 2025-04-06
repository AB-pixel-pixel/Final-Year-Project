import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

model_name = "v1"
# 定义编码器网络
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 定义强化学习模型
class ReinforcementLearningModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(ReinforcementLearningModel, self).__init__()
        #self.frame_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        #self.agent_id_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.state_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.task_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.action_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        #self.candidate_action_encoder = Encoder(embedding_dim, hidden_dim, hidden_dim)
        self.output_layer = Encoder(hidden_dim,hidden_dim//2, 1)
    
    def forward(self, frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding): #, candidate_actions_embedding):
        frame_encoded = self.state_encoder(frame_embedding)
        agent_id_encoded = self.state_encoder(agent_id_embedding)
        state_encoded = self.state_encoder(state_embedding)
        task_encoded = self.task_encoder(task_embedding)
        action_encoded = self.action_encoder(action_embedding)
    

        current_q_value = (frame_encoded + agent_id_encoded + state_encoded ) * action_encoded * task_encoded
        # print("current_q_value", current_q_value.shape)
        current_q_value = self.output_layer(current_q_value)

        #current_q_value = torch.sum(current_q_value, dim=1)
        #current_q_value = torch.mean(current_q_value, dim=1)
        # print("current_q_value", current_q_value.shape)
        # aggregate_q_value = self.output_layer(current_q_value)
        
        return current_q_value

# 加载编码后的数据
with open('/media/airs/BIN/graduation_design_env/encoded_data_v3.pkl', 'rb') as f:
    encoded_data = pickle.load(f)

# 转换数据格式，适应PyTorch模型
class CustomDataLoader:
    def __init__(self, data, batch_size=128, shuffle=True):
        # 初始化函数，用于初始化数据集
        self.data = []
        for i in data.values():
            self.data.extend(i)
        # print("len(self.data)", len(self.data))
        # 数据集
        self.batch_size = batch_size
        # 每个batch的大小
        self.shuffle = shuffle
        # 是否打乱数据集
        self.indices = np.arange(len(self.data)-1)
        # 创建一个索引数组，长度为数据集的长度
        if shuffle:
            np.random.shuffle(self.indices)
            # 如果打乱数据集，则随机打乱索引数组
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.indices):
            raise StopIteration
        
        batch_indices = self.indices[self.index:self.index + self.batch_size]
        batch = [(self.data[i],self.data[i+1]) for i in batch_indices]
        
        # for item in batch:
        #     print(item['frame_embedding'].shape)
        # 确保每个字段的大小一致
        frame_embedding0 =  torch.stack([torch.tensor(item[0]['frame_embedding'], dtype=torch.float32) for item in batch])
        agent_id_embedding0 = [torch.tensor(item[0]['agent_id_embedding'], dtype=torch.float32) for item in batch]
        state_embedding0 = [torch.tensor(item[0]['state_embedding'], dtype=torch.float32) for item in batch]
        task_embedding0 = [torch.tensor(item[0]['task_embedding'], dtype=torch.float32) for item in batch]
        action_embedding0 = [torch.tensor(item[0]['action_embedding'], dtype=torch.float32) for item in batch]
        # candidate_actions_embedding = [torch.tensor(item['candidate_actions_embedding']) for item in batch]
        reward0 = [torch.tensor(item[0]['reward'], dtype=torch.float32) for item in batch]


        frame_embedding1 =  torch.stack([torch.tensor(item[1]['frame_embedding'], dtype=torch.float32) for item in batch])
        agent_id_embedding1 = [torch.tensor(item[1]['agent_id_embedding'], dtype=torch.float32) for item in batch]
        state_embedding1 = [torch.tensor(item[1]['state_embedding'], dtype=torch.float32) for item in batch]
        task_embedding1 = [torch.tensor(item[1]['task_embedding'], dtype=torch.float32) for item in batch]
        action_embedding1 = [torch.tensor(item[1]['action_embedding'], dtype=torch.float32) for item in batch]
        # candidate_actions_embedding = [torch.tensor(item['candidate_actions_embedding']) for item in batch]
        reward1 = [torch.tensor(item[1]['reward'], dtype=torch.float32) for item in batch]
        
        # print("frame_embedding",frame_embedding.shape)
        self.index += self.batch_size
        
        return [{
            'frame_embedding': frame_embedding0,
            'agent_id_embedding': torch.stack(agent_id_embedding0),
            'state_embedding': torch.stack(state_embedding0),
            'task_embedding': torch.stack(task_embedding0),
            'action_embedding': torch.stack(action_embedding0),
            # 'candidate_actions_embedding': candidate_actions_embedding,
            'reward': torch.stack(reward0)
        },
        {
            'frame_embedding': frame_embedding1,
            'agent_id_embedding': torch.stack(agent_id_embedding1),
            'state_embedding': torch.stack(state_embedding1),
            'task_embedding': torch.stack(task_embedding1),
            'action_embedding': torch.stack(action_embedding1),
            # 'candidate_actions_embedding': candidate_actions_embedding,
            'reward': torch.stack(reward1)
        }]
    def reset(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)


# 设置超参数
embedding_dim = 768
hidden_dim = 64
learning_rate = 0.001
num_epochs = 100000
batch_size = 512

dataloader = CustomDataLoader(encoded_data, batch_size=batch_size, shuffle=True)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# 初始化模型、损失函数和优化器
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# 定义在线网络和目标网络
online_model = ReinforcementLearningModel(embedding_dim, hidden_dim).to(device)
target_model = ReinforcementLearningModel(embedding_dim, hidden_dim).to(device)

# 初始化目标网络的权重与在线网络相同
target_model.load_state_dict(online_model.state_dict())

criterion = nn.MSELoss()
optimizer = optim.SGD(online_model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    online_model.train()
    target_model.eval()
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
    cnt = 0
    for batch_state in progress_bar:
        cnt +=1
        optimizer.zero_grad()
        batch = batch_state[0]
        frame_embedding = batch['frame_embedding'].to(device)
        agent_id_embedding = batch['agent_id_embedding'].to(device)
        state_embedding = batch['state_embedding'].to(device)
        task_embedding = batch['task_embedding'].to(device)
        action_embedding = batch['action_embedding'].to(device)
        # candidate_actions_embedding = [ca.to(device) for ca in batch['candidate_actions_embedding']]
        reward = batch['reward'].to(device)


        next_state_batch = batch_state[1]
        frame_embedding1 = next_state_batch['frame_embedding'].to(device)
        agent_id_embedding1 = next_state_batch['agent_id_embedding'].to(device)
        state_embedding1 = next_state_batch['state_embedding'].to(device)
        task_embedding1 = next_state_batch['task_embedding'].to(device)
        action_embedding1 = next_state_batch['action_embedding'].to(device)
        # candidate_actions_embedding = [ca.to(device) for ca in batch['candidate_actions_embedding']]
        reward1 = next_state_batch['reward'].to(device)
        
        # 计算当前Q值（使用在线网络）
        current_q = online_model(frame_embedding, agent_id_embedding, state_embedding, task_embedding, action_embedding)#, candidate_actions_embedding)
        
        # 计算目标Q值（使用目标网络）
        with torch.no_grad():
            next_q = target_model(frame_embedding1, agent_id_embedding1, state_embedding1, task_embedding1, action_embedding1)#, candidate_actions_embedding)
            next_q = next_q.squeeze()
            target_q = reward + 0.99 * next_q
        
        # 计算损失并优化
        loss = criterion(current_q, target_q)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    dataloader.reset()
    # 定期更新目标网络
    if epoch % 100 == 0:  # 每2个epoch更新一次目标网络
        target_model.load_state_dict(online_model.state_dict())
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/cnt:.4f}')

# 保存模型
torch.save(online_model.state_dict(), f'{model_name}.pth')
print("模型已保存")