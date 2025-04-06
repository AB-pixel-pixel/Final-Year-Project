from pydantic import BaseModel, Field
from typing import Annotated,Union,Optional, Tuple ,List,Dict

# ------------------------------通信協議 START----------------------------

# ln -s /media/airs/BIN/code_base/cb_heavy/communication_protocol.py communication_protocol.py 

class PROMPT_Constituent_Elements(BaseModel):
    agent_name: str = ""
    current_room: str = ""
    progress_desc: str = ""
    need_replan: bool = False 
    current_task : str = ""
    available_plans: str = ""
    step_num : int = -1
    grabbed_objects: Union[list,str] = []
    discovery : dict = dict()
    
    # rooms_explored: dict[str,str] = {}
    # object_list: dict = {}
    # obj_per_room: dict = {}
    # satisfied: list = []
    # opponent_grabbed_objects: Optional[list[dict]] = []
    # opponent_last_room: Optional[str] = ""
    # dialogue_history: list[str] = []
    # action_history_desc: str = ""
    # Important atribution
    # goal_desc: str = ""


class BATCH_PROMPT_Constituent_Elements(BaseModel):
    batch_prompt_constituent_elements : list[PROMPT_Constituent_Elements]

class SceneName(BaseModel):
    scene_name : int

class Decision(BaseModel):
    decision: str = Field(description="agent's decision")
    reason: str = Field(description="reason for the decision")

class Agent_Decision(BaseModel):
    agent_id: int
    decision: str = Field(description="agent's decision")
    reason: str = Field(description="reason for the decision")

class Agent_Decisions(BaseModel):
    decisions: list[Agent_Decision]        

class Action_Options(BaseModel):
    batch_actions: dict[str,str]


class cwah_image_saved_path(BaseModel):
    path : str
# ------------------------------通信協議 END----------------------------
    


# ------------------------------ communication -----------------------
    
class Message(BaseModel):
    type : str  = Field(description="Message type. Only limited to 'question', or 'answer'.")
    content : str = Field(description="The actual content of the message.") 
    from_id : Optional[int] = Field(description="The ID of the sender.") 
    to_id : Optional[int] = Field(description="The ID of the recipient. Should be None when the message is question") 
# class Answer(Message):
#     agent_id : int = Field("the id of the agent who will get the answer ")
#     content : str = Field("The content of the answer")
class Answers(BaseModel):
    answers: list[Message] = Field(description="The actual content of the message.") 
    
class cwah_steps(BaseModel):
    # 定义一个名为cwah_steps的类，继承自BaseModel
    steps : int


import requests
import json



class PromptRequest(BaseModel):
    # 定义一个名为prompt的字符串类型变量
    prompt: str

class AIRS_LLM:
    def __init__(self, large_model_server_ip_port):
        self.large_model_server_ip_port = large_model_server_ip_port

    
    def invoke(self,text : str) -> dict:
        try:
            if isinstance(text, dict) or isinstance(text, list):
                text = json.dumps(text, indent=4, ensure_ascii=False)
            data_model = PromptRequest(prompt = text)
            response = requests.post(self.large_model_server_ip_port,json= data_model.dict(),timeout=6000)
            response.raise_for_status()  # 检查请求是否成功
            # print("Response from server:", response.json())
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"Request failed with status code {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while sending request: {e}")
            return {}
        
    def remote_request(self, text : str):
        completion = self.client.chat.completions.create(model="gpt-4o",
            messages=[{"role": "user", "content": text}])
        return completion


