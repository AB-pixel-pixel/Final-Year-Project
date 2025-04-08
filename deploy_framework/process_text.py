import re
from typing import Annotated,Union,Optional, Tuple ,List,Dict

def remove_labels(text)-> str:
    # 匹配模式为一个字母加上一个点和空格
    pattern = r'[A-Z]\.\s'
    # 使用 re.sub 去掉匹配到的部分
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text



def extract_pairs(text) -> List:
    """从文本中提取<>中的内容和后面的()中的数字，组成字典"""
    # print("extract_pairs text: ", text)
    pattern = r'<(.*?)>\s*\((\d+)\)'
    try:
        matches = re.findall(pattern, text)
        # print("extract_pairs matches: ", matches)
    except TypeError as e:
        print("TypeError", e)
        return []
    return [(key, value) for key,value in matches]

def get_name_to_id(name:str) -> int:
    name_to_id_dict = {
        "Alice":0,
        "Bob":1
    }
    return name_to_id_dict[name]


def get_id_to_name(id: int) -> Optional[str]:
    id_to_name_dict = {
        0: "Alice",
        1: "Bob"
    }
    return id_to_name_dict.get(id)


def convert_discovery_to_text(data):
    result = []
    ans = ""
    for room, items in data.items():
        temp = []
        for item in items:
            # Extract the item name and ID
            objects = extract_pairs(item)
            if objects == []:
                continue
            for placeholder0, placeholder1 in objects:
                if placeholder0.isdigit():
                    temp.append(f"<{placeholder1.strip()}> ({placeholder0})")
                else:
                    temp.append(f"<{placeholder0.strip()}> ({placeholder1})")
        if temp != []:
            objects_in_same_room = ",".join(temp)
            if len(temp) > 1:
                objects_in_same_room += " are"
            else:
                objects_in_same_room += " is"
            objects_in_same_room += f" in {room}."
            result.append(objects_in_same_room)
            ans = f"The distribution of objects in the rooms is: {''.join(result)} "
        else:
            ans = "Observation nothing. "
    return ans



