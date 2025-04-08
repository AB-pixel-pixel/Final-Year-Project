from langchain_community.llms import Ollama
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
import requests
import numpy as np
import random 
import re
from LOG_SYSTEM import LOG_SYSTEM, log_system
from config import *
from framework_structure import *
def is_replan(target_step:str) -> bool:
    # 将目标步骤转换为小写
    _target_step_ = target_step.lower()
    return "replan" in _target_step_ or "if" in _target_step_
    
def fix_error_type_2(target_step,options: str) ->str:
    options = [option for option in options.split("\n") if option.strip() ]
    rooms = ["bedroom", "kitchen", "living room", "bathroom", "office"]
    target_step = target_step.lower()
    for room in rooms:
        if room in target_step:
            for option in options:
                if room in option:
                    return option

def change_available_plans(available_plans : str) -> str:
    available_plans = available_plans.replace('\n','')
    available_plans = available_plans.replace('A. send a message:','')
    available_plans = available_plans.replace('B. ','A. ')
    available_plans = available_plans.replace('C. ','B. ')
    available_plans = available_plans.replace('D. ','C. ')
    available_plans = available_plans.replace('E. ','D. ')
    available_plans = available_plans.replace('F. ','E. ')
    available_plans = available_plans.replace('G. ','F. ')
    available_plans = available_plans.replace('H. ','G. ')
    available_plans = available_plans.replace('I. ','H. ')
    available_plans = available_plans.replace('J. ','I. ')
    available_plans = available_plans.replace('K. ','J. ')
    available_plans = available_plans.replace('L. ','K. ')
    available_plans = available_plans.replace('M. ','L. ')
    available_plans = available_plans.replace('N. ','M. ')
    available_plans = available_plans.replace('O. ','N. ')
    available_plans = available_plans.replace('P. ','O. ')
    available_plans = available_plans.replace('Q. ','P. ')
    available_plans = available_plans.replace('R. ','Q. ')
    available_plans = available_plans.replace('S. ','R. ')
    available_plans = available_plans.replace('T. ','S. ')
    available_plans = available_plans.replace('U. ','T. ')
    available_plans = available_plans.replace('V. ','U. ')
    available_plans = available_plans.replace('W. ','V. ')
    available_plans = available_plans.replace('X. ','W. ')
    available_plans = available_plans.replace('Y. ','X. ')
    available_plans = available_plans.replace('Z. ','Y. ')
    return available_plans

def reverse_change_decisions(temp : "Decision") -> str:
    log_system.PRINT("reverse_change_decisions")
    temp_text :str = temp.decision
    temp_text = temp_text.replace('Y. ', 'Z. ')
    temp_text = temp_text.replace('X. ', 'Y. ')
    temp_text = temp_text.replace('W. ', 'X. ')
    temp_text = temp_text.replace('V. ', 'W. ')
    temp_text = temp_text.replace('U. ', 'V. ')
    temp_text = temp_text.replace('T. ', 'U. ')
    temp_text = temp_text.replace('S. ', 'T. ')
    temp_text = temp_text.replace('R. ', 'S. ')
    temp_text = temp_text.replace('Q. ', 'R. ')
    temp_text = temp_text.replace('P. ', 'Q. ')
    temp_text = temp_text.replace('O. ', 'P. ')
    temp_text = temp_text.replace('N. ', 'O. ')
    temp_text = temp_text.replace('M. ', 'N. ')
    temp_text = temp_text.replace('L. ', 'M. ')
    temp_text = temp_text.replace('K. ', 'L. ')
    temp_text = temp_text.replace('J. ', 'K. ')
    temp_text = temp_text.replace('I. ', 'J. ')
    temp_text = temp_text.replace('H. ', 'I. ')
    temp_text = temp_text.replace('G. ', 'H. ')
    temp_text = temp_text.replace('F. ', 'G. ')
    temp_text = temp_text.replace('E. ', 'F. ')
    temp_text = temp_text.replace('D. ', 'E. ')
    temp_text = temp_text.replace('C. ', 'D. ')
    temp_text = temp_text.replace('B. ', 'C. ')
    temp_text = temp_text.replace('A. ', 'B. ')
    temp.decision = temp_text
    return temp


def get_step2_action_prompt(action_stystem_prompt,adaptive_prompt,desc_o_s,available_plans,target_step):
    # 初始化提示信息列表
    prompts = []

    # 根據變量添加文字介紹
    if available_plans:
        if "send a message" in available_plans:
            prompts.append(f"Multiple choice questions to be completed: {change_available_plans(available_plans)}")
        else:
            prompts.append(f"Multiple choice questions to be completed: {available_plans}")
    if desc_o_s:
        prompts.append(f"Observation: {desc_o_s}")

    if adaptive_prompt:
        prompts.append(f"Rules: {adaptive_prompt}")

    if target_step:
        prompts.append(f"Our target is: {target_step}")
    
    if action_stystem_prompt:
        ans = action_stystem_prompt+'.'.join(prompts)
    else: 
        raise ValueError("system_prompt must not be empty")

    return ans





def step2action_by_similarity(target_step,available_plans:str):

    def remove_labels(text):
        # 匹配模式为一个字母加上一个点和空格
        pattern = r'[A-Z]\.\s'
        # 使用 re.sub 去掉匹配到的部分
        cleaned_text = re.sub(pattern, '', text)
        return cleaned_text
    

    cleaned_available_plans = remove_labels(available_plans)
    options = [option for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option ]  
    most_similar_sentence,max_similarity,max_index = compare_sentence_with_list(target_step,options)
    origin_options = [option for option in available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
    selected_sentence = origin_options[max_index]
    return selected_sentence,max_similarity





# 使用检索的方法來獲取step2action

def get_sentence_embedding(sentence):
    """获取句子的嵌入向量"""
    response = requests.post('http://localhost:11434/api/embed', json={
        "model": "llama3:latest",
        "input": sentence
    },timeout=60)
    embedding = response.json().get('embeddings', [])
    # log_system.PRINT(response)
    return np.array(embedding)

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    x1 = np.squeeze(np.linalg.norm(vec1), axis=0)
    x2 = np.squeeze(np.linalg.norm(vec2), axis=0)
    x3 = vec1.dot(vec2.T)
    ans = x3 / (x1 * x2)
    return ans
    
def calculate_bigram_overlap(sentence1, sentence2):
    """计算两个句子的2-gram重叠度"""
    def ngrams(sentence, n):
        words = sentence.split()
        return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]

    bigrams1 = set(ngrams(sentence1, 2))
    bigrams2 = set(ngrams(sentence2, 2))

    intersection = bigrams1.intersection(bigrams2)
    union = bigrams1.union(bigrams2)

    return len(intersection) / len(union) if len(union) > 0 else 0, intersection, union

def jaccard_similarity(str1, str2):
    # 将字符串分割成单词并转换为集合
    set1 = set(str1.split())
    set2 = set(str2.split())
    
    # 计算交集和并集
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    
    # 计算 Jaccard 相似度
    if len(union) == 0:  # 防止除以零
        return 0.0
    return len(intersection) / len(union)

def compare_sentence_with_list(sentence, sentence_list_):
    """比较一句话与句子列表的相似度"""
    # 获取目标句子的嵌入
    target_embedding = get_sentence_embedding(sentence)

    most_similar_sentence = None
    max_similarity = -np.inf
    max_index = None
    # similarities = []
    
    log_system.PL("---------------相似度动作选择-------------------")
    log_system.PL(f"Input: {sentence}")
    sentence = sentence.lower()
    for index,s in enumerate(sentence_list_):
        s = s.lower()
        s = s.replace("<", "")
        s = s.replace(">", "")
        # 获取列表中每个句子的嵌入
        list_embedding = get_sentence_embedding(s)
        # 计算余弦相似度
        similarity = cosine_similarity(target_embedding, list_embedding)
        # 计算2-gram重叠度

        jaccard = jaccard_similarity(sentence, s)

        # 计算平均值
        average_similarity = (similarity + jaccard) / 2
        
        log_system.LOG(f"Option {index}: {s} \t Cosine Similarity: {similarity} \t jaccard_similarity: {jaccard} \t Average Similarity: {average_similarity}")
        # similarities.append(similarity)
        # 更新最大相似度和对应句子
        if average_similarity > max_similarity:
            max_similarity = average_similarity
            most_similar_sentence = s
            max_index = index
    log_system.PL(f"Input: {sentence}\n最佳匹配的句子： {most_similar_sentence}")

    return most_similar_sentence,max_similarity,max_index


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_similarity(input_sentence, options):
    # todo output similarity > 1
    # 包括输入和选项
    sentences = [input_sentence] + options

    # 词频矩阵
    vectorizer = CountVectorizer().fit_transform(sentences)
    vectors = vectorizer.toarray()

    # 计算余弦相似度
    cosine_sim = cosine_similarity(vectors)
    
    # 获取与输入句子的相似度
    return cosine_sim[0][1:]



def wrapper_similarity_v4(input_sentence, options):
    # 计算相似度
    similarities = calculate_similarity(input_sentence, options)

    # 找到最佳匹配
    best_match_index = similarities.argmax()
    best_match_sentence = options[best_match_index]
    best_similarity = similarities[best_match_index]

    # 创建结果字典
    results = {options[i]: (round(similarities[i],2),i) for i in range(len(options))}
    # print("results",results)
    filtered_sorted_results = sorted(
    ((option, similarity[0],similarity[1]) for option, similarity in results.items() if similarity[0] > 0),
    key=lambda x: x[1],
    reverse=True
    )
    # 展示结果
    if display_inference_process:
        log_system.PL(f"input_sentence: {input_sentence}")
        log_system.PL(f"Best Match: {best_match_sentence} (Similarity: {best_similarity})")
        log_system.PL("All Options and Their Similarities:")
        for sentence, score in results.items():
            log_system.PL(f"{sentence}: {score}")
    else:
        log_system.LOG(f"input_sentence: {input_sentence}")
        log_system.LOG(f"Best Match: {best_match_sentence} (Similarity: {best_similarity})")
        log_system.LOG("All Options and Their Similarities:")
        for sentence, score in results.items():
            log_system.LOG(f"{sentence}: {score}")

    return filtered_sorted_results, best_match_index





_rooms = ["bedroom", "kitchen", "living room","livingroom", "bathroom", "office","bed"]
_public_object = ['bowl', 'plate', 'tea_tray', 'plastic_basket', 'wicker_basket', 'wood_basket', 'bread', 'burger', 'loaf_bread', 'apple', 'banana', 'orange', 'iphone', 'pen', 'key', 'ipod', 'lighter', 'purse', 'calculator', 'pencil_bucket', 'mouse']





#  matching



def align_name_id(_target_step,_most_similar_sentence) -> int:
    ids0 = re.findall(r'\d+', _target_step)
    ids1 = re.findall(r'\d+', _most_similar_sentence)
    # subjects1, verbs1,ids1 = get_subject_verb(_most_similar_sentence)

    for target_object in _public_object:
        if target_object in _target_step and target_object not in _most_similar_sentence:
            return 2
        
    for room in _rooms:
        if room in _target_step and room not in _most_similar_sentence:
            return 2
        
    if len(ids0) != 0:
        for id0 in ids0:
            for id1 in ids1:
                if id0 == id1:
                    return 0
    else:
        return 0

def verify_based_on_rules(target_step,most_similar_sentence) -> int:
    """ 
    基于规则判断是否有错误匹配的情况存在,
    1: action mismatch error : grasp对应错误的go to
    2: object type error : go to 的房间，对应错误
    3: grasp action object error: unaligned
    4: 不符合任何模式的错误
    """


    _target_step =  target_step
    _most_similar_sentence = most_similar_sentence.lower()
    
    # 当目标动作是抓取，匹配的动作也应该和抓取有关
    if "grasp" in _target_step or "grab" in _target_step:
        if "grasp" in _most_similar_sentence:
            # 直接锁定名字
            for public_obj in _public_object:
                if public_obj in _target_step and public_obj in _most_similar_sentence:
                    return 0
            # 视没有相同名字

            # action align success, align object type
            # if "container" in _target_step and "container" in _most_similar_sentence:
            #     return 0
            if "container" in _target_step and "container" not in _most_similar_sentence:
                return 3

            # make sure the object is the same
            return align_name_id(_target_step,_most_similar_sentence)
            
        
        elif "put" in _most_similar_sentence:
            return 0
        else:
            return 1
        
    if "put" in _target_step:
        if "go to" in _most_similar_sentence or "transport" in _most_similar_sentence:
            return 1
        elif "put" in _most_similar_sentence:
            # match the id and object
            return align_name_id(_target_step, _most_similar_sentence)

    if "pick" in _target_step:
        if "grasp" in _most_similar_sentence or "put" in _most_similar_sentence:
            return 0
        # elif "transport" in _most_similar_sentence:
        #     return 1
        else:
            return 1
    # 非常强的约束           
    if "transport" in _target_step and "bed" in _target_step:
        if "bedroom" in _most_similar_sentence or "bed" in _most_similar_sentence:
            return 0
        else:
            return 2
    
    if ("go to" in _target_step or "explore" in _target_step or "exploring" in _target_step ) and ("go to" in _most_similar_sentence or "explore" in _most_similar_sentence):
        # 当目标动作是移动，则移动的地方和房间应该和选项中的房间是相关的，编号也应该是匹配的
        
        # 到了这里，基本可以判定为走的房间至少是一样的了，接下来检查编号
        print("_target_step",_target_step,"_most_similar_sentence",_most_similar_sentence)
        return align_name_id(_target_step, _most_similar_sentence)

    return 4

def find_same_id_option(target_id,options) -> str:
    for option in options:
        match2 = re.search(r'\((\d+)\)', option)
        if match2:
            if target_id == match2.group(1):
                return option
    return None

def fix_error_type_3(target_step : str,options : list[str]) -> str:
    """ 
    修正错误匹配的情况:
    如果出现编号错误,则return  None
    """
    target_id = None
    _target_step =  target_step.lower()
    match1 = re.search(r'\((\d+)\)', _target_step)
    if match1:
        target_id = match1.group(1)

    # 房间对齐
    rooms = ["bedroom", "kitchen", "living room", "bathroom", "office","bed"]
    for room in rooms:
        for option in options:
            if room in _target_step and room in option:
                match2 = re.search(r'\((\d+)\)', option)
                # 编号对齐
                if match2 and target_id:
                    if target_id == match2.group(1):
                        return option
                if room == "bed":
                    return option
    # 如果到了这一步，说明了编号匹配失败，原选项中没有的真正匹配的选项
    return ""            


def random_select(available_plans:str) -> str:
    origin_options = [option for option in available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
    if origin_options:  # 确保列表不为空
        return random.choice(origin_options)
    return ""  # 如果没有可选项，返回 None

def select_based_on_word_frequency(target_step:Optional[str], available_plans:str) -> str:
    """ based on work frequency to choose the optino """
    if target_step is not None:
        cleaned_available_plans = remove_labels(available_plans)
        target_step = target_step.lower().replace("<","").replace(">","").replace("_"," ")
        options = [option.lower().replace("<","").replace(">","").replace("_"," ") for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option ]  
        filtered_sorted_results,max_index = wrapper_similarity_v4(target_step,options)
        origin_options = [option for option in available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
        selected_sentence = origin_options[max_index]
        return selected_sentence  # 如果没有可选项，返回 None
    return ""

def select_based_on_word_frequency_v1(target_step:Optional[str], available_plans:str) -> str:
    """ based on work frequency to choose the optino """
    if target_step is not None:
        cleaned_available_plans = remove_labels(available_plans)
        target_step = target_step.lower().replace("<","").replace(">","").replace("_"," ").replace("to","")
        options = [option.lower().replace("<","").replace(">","").replace("_"," ").replace("to","") for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option ]  
        filtered_sorted_results,max_index = wrapper_similarity_v4(target_step,options)
        origin_options = [option for option in available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
        selected_sentence = origin_options[max_index]
        return selected_sentence  # 如果没有可选项，返回 None
    return ""

def step2action_by_similarity_v4(target_step,available_plans:str):
    """ 
    传统方法进行匹配，再基于人工编写的规则进行排查和重新匹配。
    return 清理过后的句子，相似度，是否重新规划，原句
    若无法找到正确的匹配结果，就返回None,None,True

    action alignment : 
    "go to" == "explore"
    "grasp" == "grab"

    """
    # TODO: 修改这个匹配方法为纯粹的字符串匹配或者添加模糊匹配等规则
    # print("available_plans",available_plans)
    if  is_replan(target_step):
        log_system.PL("Info : this step is calling replan")
        return None,None,True,None
    
    cleaned_available_plans = remove_labels(available_plans)
    target_step = target_step.lower().replace("<","").replace(">","").replace("_"," ").strip()
    options = [option.lower().replace("<","").replace(">","").replace("_"," ").strip() for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option ]  
    filtered_sorted_results,max_index = wrapper_similarity_v4(target_step,options)
    origin_options = [option for option in available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
    selected_sentence = origin_options[max_index]
    
    error_type = None
    for selected_sentence,similarity_value,original_index in filtered_sorted_results:
        # 直接保送
        if similarity_value >= 0.98:
            return origin_options[original_index],similarity_value,False, selected_sentence
        
        error_type = verify_based_on_rules(target_step,selected_sentence)
        if error_type == 0:
            return origin_options[original_index],similarity_value,False, selected_sentence
        elif error_type == 1:
            log_system.PL("Warn: 此次匹配出现了动作匹配错误的情况 selected_sentence:\t" ,selected_sentence)
        elif error_type == 2:
            log_system.PL("Warn: 此次匹配出现target object 不匹配的情况 selected_sentence:\t" ,selected_sentence)
            # fix_error_type_2(target_step,selected_sentence,options)
        elif error_type == 3:
            log_system.PL(f"Warn: object id mismatch,将返回None: {selected_sentence}")
        # else:
        #     log_system.PL(f"Error: there will be code mistake in the step2action matching:{selected_sentence}")
    if error_type:
        if error_type != 0:
            log_system.PL(f"Error: there will be code mistake in the step2action matching:{selected_sentence}")
    return None,None,False,None # 这里False的原因是，不是主动请求replan的

def step2action_by_similarity_v5(target_step,available_plans:str):
    """ 
    传统方法进行匹配，再基于人工编写的规则进行排查和重新匹配。
    return 清理过后的句子，相似度，是否重新规划，原句
    若无法找到正确的匹配结果，就返回None,None,True

    action alignment : 
    "go to" == "explore"
    "grasp" == "grab"

    """
    # TODO: 修改这个匹配方法为纯粹的字符串匹配或者添加模糊匹配等规则
    # print("available_plans",available_plans)
    if  is_replan(target_step):
        log_system.PL(f"Info : this step : {target_step} is calling replan.")
        return None,None,True,""
    
    cleaned_available_plans = remove_labels(available_plans)
    target_step = target_step.lower().replace("<","").replace(">","").replace("_"," ") 
    options = [option.lower().replace("<","").replace(">","").replace("_"," ") for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option ]  
    filtered_sorted_results,max_index = wrapper_similarity_v4(target_step,options)
    origin_options = [option for option in available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
    cleared_options = [option for option in cleaned_available_plans.split("\n") if option.strip() and not "A. send a message" in option ]
    selected_sentence = origin_options[max_index]
    
    error_type = None
    for selected_sentence,similarity_value,original_index in filtered_sorted_results:
        # 直接保送
        if similarity_value >= 1.0:
            return origin_options[original_index],similarity_value,False, cleared_options[original_index]
    return None,None,False,"" # 这里False的原因是，不是主动请求replan的

def clean_available_plan(available_plans : str) -> List:
    cleaned_available_plans = remove_labels(available_plans)
    options = [option.lower().replace("<","").replace(">","").replace("_"," ") for option in cleaned_available_plans.split("\n") if option.strip()]  
    


def request_llama3(prompt):
    temp_model = Ollama(model="llama3")
    res = temp_model.invoke(prompt)
    return res


if __name__ == '__main__':
    # 示例用法
    sentence_A = 'go grasp target object apple' 
    result_str = "A. go grasp target object banana \n B. explore current room livingroom (4000)\n C. go to livingroom (1000) \n D. go to bedroom (8000) \n E. go to office (1000)"    
    selected_sentence, max_similarity = step2action_by_similarity_v4(sentence_A, result_str)
    if selected_sentence and max_similarity:
        log_system.PRINT(f"Sentence_A: {sentence_A}")
        log_system.PRINT(f"Most similar sentence: {selected_sentence}")
        log_system.PRINT(f"Max similarity: {max_similarity}")
    else:
        log_system.PRINT("空缺值")