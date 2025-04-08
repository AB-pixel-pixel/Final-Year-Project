import pandas as pd
import glob
import os
import re


# 读取单个 CSV 文件并计算按 Token Source 分组的总消耗量
def summarize_single_file(file_path):
    data = pd.read_csv(file_path)
    token_summary = data.groupby('Token Source')['Increment'].sum().reset_index()
    print("token_summary",token_summary)
    total_tokens = token_summary['Increment'].sum()
    token_summary['Percentage'] = round(token_summary['Increment'] / total_tokens * 100,3)
    percentage_dict = dict(zip(token_summary['Token Source'], token_summary['Percentage']))
    return percentage_dict

# 定义路径和文件模式
path = '/home/airs/bin/ai2thor_env/cb/logs_cwah_token_usage'  # 替换为你的文件路径
file_pattern = os.path.join(path, '*v1*token_usage_log.csv')

# path = "/media/airs/BIN/baseline/Co-LLM-Agents/tdw_mat/logs"
# file_pattern = os.path.join(path, '*token_usage_log.csv')

# 获取所有符合条件的文件
files = glob.glob(file_pattern)

# 初始化总和和计数器
total_prompt_tokens = 0
total_completion_tokens = 0
total_tokens = 0
count = 0
total_cost = 0
# List to store results for CSV
results = []

# 遍历每个文件并计算总和
for file in files:
    df = pd.read_csv(file)
    

    token_summary = summarize_single_file(file)
    # print(token_summary)

    # 计算当前文件的平均值
    current_prompt_avg = df['Prompt Tokens'].sum()
    current_completion_avg = df['Completion Tokens'].sum()
    
    # 输出当前文件的平均值
    print(f'File: {os.path.basename(file)}')
    print(f'Total Prompt Tokens: {current_prompt_avg:.2f}')
    print(f'Total Completion Tokens: {current_completion_avg:.2f}')

    
    # 累加总和
    total_prompt_tokens += df['Prompt Tokens'].sum()
    total_completion_tokens += df['Completion Tokens'].sum()
    
    # 仅取最后一行的 Total Tokens
    if not df.empty:
        print("total_tokens", df['Total Tokens'].iloc[-1])
        total_tokens += df['Total Tokens'].iloc[-1]  # 获取最后一行的 Total Tokens

    if not df.empty:
        cost = round(df[' Cost'].iloc[-1],3)
        print("cost", cost)
        total_cost += cost
    count += 1
    print()
    # Append results for this file to the results list
    
    match = re.search(r'scene(\d+)', file)
    if match:
        scene_name = match.group(1)
        temp = {
            'File': scene_name,
            'Prompt Tokens': current_prompt_avg}
    
        temp.update(token_summary)
        temp.update({'Completion Tokens': current_completion_avg,
            'Tokens': current_prompt_avg + current_completion_avg if not df.empty else 0,
            'Cost':  cost})
        results.append(temp)
    else:
        print("Error: can't extract scene name in file path")

# 计算整体平均值
average_prompt_tokens = total_prompt_tokens / count if count > 0 else 0
average_completion_tokens = total_completion_tokens / count if count > 0 else 0
average_total_tokens = average_prompt_tokens + average_completion_tokens # total_tokens / count if count else 0  # 以文件数计算整体平均

# 输出整体结果
print(f'\nOverall Average Prompt Tokens: {average_prompt_tokens:.2f}')
print(f'Overall Average Completion Tokens: {average_completion_tokens:.2f}')
print(f'Overall Average Total Tokens: {average_total_tokens:.2f}')

results.append({
    'File': "Average",
    'Prompt Tokens': average_prompt_tokens,
    'Completion Tokens': average_completion_tokens,
    'Tokens': average_total_tokens if not df.empty else 0,
    'Cost' :  total_cost  / count if count > 0 else 0
})
results.append({
    'File': "Total",
    'Prompt Tokens': total_prompt_tokens,
    'Completion Tokens': total_completion_tokens,
    'Tokens': total_completion_tokens + total_prompt_tokens if not df.empty else 0,
    'Cost' :  total_cost
})


# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(path, 'token_usage_summary.csv'), index=False)

print(f'Results saved to {os.path.join(path, "token_usage_summary.csv")}')