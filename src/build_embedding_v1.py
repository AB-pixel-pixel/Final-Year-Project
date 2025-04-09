# -*- coding: utf-8 -*-

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm

# 加载预训练模型和分词器
model_name_or_path = '/media/airs/BIN/LargeModels/embedding_models'
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True)
dimension = 768

# 加载数据
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
# /media/airs/BIN/tdw_ex/training_data/try/LMs-collect_train_dataset_v1/eval_result.json
training_data = load_pickle("/media/airs/BIN/graduation_design_env/all_scenes_data_v5.pkl")

# 定义函数，将文本转换为embedding
def get_embedding(texts):
    # print(type(texts),'\n',texts)
    if not isinstance(texts, str):
        texts = str(texts)
    # 对文本进行分词和编码
    batch_dict = tokenizer(texts, max_length=4000, padding=True, truncation=True, return_tensors='pt')
    # 使用模型进行编码
    outputs = model(**batch_dict)
    # 提取第一个标记的嵌入向量作为整个文本的嵌入表示
    embeddings = outputs.last_hidden_state[:, 0]
    # 对嵌入向量进行归一化
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

# 对每个元组中的每个属性进行编码
encoded_data = dict()
scene_counter = 0
for scene in tqdm(training_data, desc="Processing Scenes"):
    encoded_data[scene_counter] = []

    for item in tqdm(scene, desc="Processing Items", leave=False):
        # 分别对每个文本属性进行编码
        frame_embedding = get_embedding(item[0])  # Frame 描述
        agent_id_embedding = get_embedding(item[1])  # Agent Id 描述
        state_embedding = get_embedding(item[2])  # 状态描述
        task_embedding = get_embedding(item[3])   # 任务描述
        action_embedding = get_embedding(item[4]) # 当前动作
        
        # 对候选动作列表中的每个候选动作单独编码
        # candidate_actions_embedding = []
        # for candidate_action in item[5]:
        #     ca_embedding = get_embedding(candidate_action)
        #     candidate_actions_embedding.append(ca_embedding)
        
        reward = item[6]                            # 奖励

        # 将编码后的属性和奖励保存到新的数据结构中
        encoded_item = {
            'frame_embedding' : frame_embedding.detach().numpy(),
            'agent_id_embedding': agent_id_embedding.detach().numpy(),
            'state_embedding': state_embedding.detach().numpy(),
            'task_embedding': task_embedding.detach().numpy(),
            'action_embedding': action_embedding.detach().numpy(),
            'candidate_actions_embedding': [],
#            'candidate_actions_embedding': [ca.detach().numpy() for ca in candidate_actions_embedding],
            'reward': reward
        }
        encoded_data[scene_counter].append(encoded_item)
    scene_counter +=1


# 保存编码后的数据
saving_file = './encoded_data_v4.pkl'
with open(saving_file, 'wb') as f:
    pickle.dump(encoded_data, f)

print(f"编码完成并保存到 {saving_file} 文件中")