import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import re
import warnings
warnings.filterwarnings("ignore")
import os 


#读取词记表现
data_path = '/home/houwanlong/ad_project/data_pipeline/data'
file_name = 'source_data_noconfig.csv'
file_path = data_path + '/' + file_name

source_data = pd.read_csv(file_path, encoding='utf-8')

source_data['利润'] = source_data['原价GTV(单位元)'] - source_data['点击量'] * source_data['点击单价(单位元)']

# 对同一个关键词同一天进行聚合
source_data['总消费_temp'] = source_data['千次展示消费(单位元)'] * source_data['曝光量'] / 1000

# 定义分组列（仅保留日期和关键词）
group_cols = ['推广日期', '关键词']

# 分组聚合（仅保留必要指标）
aggregated_df = (
    source_data
    .groupby(group_cols)
    .agg({
        '曝光量': 'sum',
        '点击量': 'sum',
        '转化销量': 'sum',
        '原价GTV(单位元)': 'sum',
        '总消费_temp': 'sum',
        '利润':'sum'
    })
    .reset_index()
)

# 向量化计算衍生指标
aggregated_df = aggregated_df.assign(
    点击率=lambda x: (x['点击量'] / x['曝光量'].replace(0, 1) * 100),
    千次展示消费=lambda x: (x['总消费_temp'] / x['曝光量'].replace(0, 1) * 1000),
    点击单价=lambda x: x['总消费_temp'] / x['点击量'].replace(0, 1),
    转化率=lambda x: (x['转化销量'] / x['点击量'].replace(0, 1) * 100)  # 新增转化率
).fillna(0).round(2)

# 重命名并选择最终列
wordmetric_df = aggregated_df[[
    '推广日期', 
    '关键词',
    '曝光量',
    '点击量',
    '点击率',
    '千次展示消费',
    '转化销量',
    '点击单价',
    '原价GTV(单位元)',
    '转化率',
    '利润'  # 新增列
]].rename(columns={
    '曝光量': '总曝光量',
    '点击量': '总点击量',
    '点击率': '平均点击率(*100%)',
    '千次展示消费': '平均千次展示消费(单位元)',
    '转化销量': '总转化销量',
    '点击单价': '平均点击单价(单位元)',
    '原价GTV(单位元)': '总原价GTV(单位元)',
    '转化率': '平均转化率(*100%)',
    '利润': '总利润'  # 新增重命名
})

# 创建临时列计算总消费
source_data['总消费_temp'] = source_data['千次展示消费(单位元)'] * source_data['曝光量'] / 1000

# 定义分组列（去除城市列）
group_cols = ['推广日期', '推广计划id', '推广计划名称', '关键词']

# 进行分组聚合
aggregated_df = source_data.groupby(group_cols).agg({
    '曝光量': 'sum',
    '点击量': 'sum',
    '转化销量': 'sum',
    '原价GTV(单位元)': 'sum',
    '总消费_temp': 'sum',
    '利润': 'sum'
}).reset_index()

# 计算聚合后的点击率、千次展示消费和点击单价
aggregated_df['点击率(*100%)'] = aggregated_df.apply(
    lambda x: (x['点击量'] / x['曝光量'] * 100) if x['曝光量'] != 0 else 0,
    axis=1
)
aggregated_df['千次展示消费(单位元)'] = aggregated_df.apply(
    lambda x: (x['总消费_temp'] / x['曝光量'] * 1000) if x['曝光量'] != 0 else 0,
    axis=1
)
aggregated_df['点击单价(单位元)'] = aggregated_df.apply(
    lambda x: (x['总消费_temp'] / x['点击量']) if x['点击量'] != 0 else 0,
    axis=1
)
aggregated_df['转化率(*100%)'] = aggregated_df.apply(
    lambda x: (x['转化销量'] / (x['点击量'] if x['点击量'] != 0 else 1) * 100),
    axis=1
)

# 删除临时列
aggregated_df.drop(columns=['总消费_temp'], inplace=True)

# 调整列顺序以确保正确性（去除城市列）
final_columns = [
    '推广日期', '推广计划id', '推广计划名称', '关键词', '利润',
    '曝光量', '点击量', '点击率(*100%)', '千次展示消费(单位元)', '转化销量', '转化率(*100%)', '点击单价(单位元)', '原价GTV(单位元)'
]
source_data = aggregated_df[final_columns]

source_data = pd.merge(
    source_data,
    wordmetric_df,
    on=["推广日期", "关键词"],
    how="left"  # or 'left', 'right', 'outer' depending on your needs
)

df_0 = source_data[['推广日期','推广计划名称','关键词','利润','曝光量','点击量','点击率(*100%)','转化率(*100%)','点击单价(单位元)','原价GTV(单位元)']]

# 计算每一个关键词的PV：未来利润的现值，the present value of future profit

# Ensure the date is in datetime format
df_0['推广日期'] = pd.to_datetime(df_0['推广日期'])

# Sort the DataFrame by promotion plan, keyword, and date
df_0 = df_0.sort_values(['推广计划名称', '关键词', '推广日期'])

# Define the discount rate (e.g., 10%)
discount_rate = 0.10

# Calculate the label: present value of future profits for each promotion plan-keyword-date combination
df_0['PV'] = df_0.groupby(['推广计划名称', '关键词']).apply(
    lambda group: group.sort_values('推广日期').apply(
        lambda row: (
            group.loc[
                group['推广日期'] >= row['推广日期'], '利润'
            ].div(
                (1 + discount_rate) ** (
                    (group.loc[
                        group['推广日期'] >= row['推广日期'], '推广日期'
                    ] - row['推广日期']).dt.days / 365
                )
            )
        ).sum(), axis=1
    )
).reset_index(drop=True)

# Get the next record profit as label
df_0['利润_s1'] = df_0.groupby(['推广计划名称', '关键词'])['利润'].shift(-1)

def safe_divide(a, b):
    """安全除法，处理除零和无效值"""
    return np.divide(a, b, out=np.zeros_like(a), where=(b != 0) & (~np.isnan(b)))

# 预处理数据
df_0['点击单价(单位元)'] = df_0['点击单价(单位元)'].replace(0, np.nan).clip(lower=0.0001)
df_0['当日效率比'] = safe_divide(df_0['曝光量'], df_0['点击单价(单位元)'])

# 填充缺失值（使用前向填充）
df_0['当日效率比'] = df_0.groupby(['推广计划名称', '关键词'])['当日效率比'].ffill()

def calculate_pv(group):
    """计算每个分组的现值"""
    group = group.sort_values('推广日期')
    pv_values = []
    
    for idx, row in group.iterrows():
        # 获取未来日期数据（包括当天）
        future_mask = (group['推广日期'] >= row['推广日期'])
        future_data = group.loc[future_mask, ['推广日期', '当日效率比']]
        
        # 计算时间差（年）
        time_diff = (future_data['推广日期'] - row['推广日期']).dt.days / 365
        
        # 计算折现因子
        discount_factors = (1 + discount_rate) ** time_diff
        
        # 计算现值
        discounted_efficiency = future_data['当日效率比'] / discount_factors.values
        pv = discounted_efficiency.sum()
        
        pv_values.append(pv)
    
    group['流量CPC效率比_PV'] = pv_values
    return group

# 分组计算现值
df_0 = df_0.groupby(['推广计划名称', '关键词'], group_keys=False).apply(calculate_pv)

all_date = df_0['推广日期'].unique().tolist()

train_date = all_date[:int(len(all_date) / 5 * 1)]
test_date = all_date[int(len(all_date) / 5 * 1):]

train_df = df_0[df_0['推广日期'].isin(train_date)]
test_df = df_0[df_0['推广日期'].isin(test_date)]

dfs_byName = test_df.groupby('推广计划名称')

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
import pandas as pd

# 指定字体路径
font_path = '/home/houwanlong/ad_project/agents/SimHei.ttf'
font = FontProperties(fname=font_path, size=14)

# Initialize models and their names
topk = 7
reward_type='利润' #流量CPC效率比_PV
llm = 'gpt-4.1-nano'

from model import *

benchmark = Model_Benchmark(topk=topk, reward_type=reward_type)
model_kpmagent = Model_KPMAgent(topk=topk, reward_type=reward_type, llm=llm, seed = 42)
model_click = Model_Click(topk=topk, reward_type=reward_type)
model_impression = Model_Impression(topk=topk, reward_type=reward_type)
model_ctr = Model_CTR(topk=topk, reward_type=reward_type)
model_cvr = Model_CVR(topk=topk, reward_type=reward_type)
model_trendimpression = Model_TrendImpression(topk=topk, reward_type=reward_type)

models = [model_click, model_impression, model_ctr, model_cvr, model_trendimpression, model_kpmagent]
model_names = ['Click', 'Impression', 'CTR', 'CVR', 'Trend_Impression', 'KPMAgent']

results = []

# for name, df_byName in tqdm(list(dfs_byName)[:3], desc="Processing first 3 groups"):
for name, df_byName in tqdm(dfs_byName):
    #去除只有少数关键词的计划
    if len(df_byName['关键词'].unique())<10:
        continue

    df_byName = df_byName.copy()
    df_byName['推广日期'] = pd.to_datetime(df_byName['推广日期'])
    df_sorted = df_byName.sort_values('推广日期')
    
    min_date = df_sorted['推广日期'].min()
    max_date = df_sorted['推广日期'].max()
    start_dates = pd.date_range(start=min_date, end=max_date - pd.Timedelta(days=6), freq='D')
    
    for start in start_dates:
        end = start + pd.Timedelta(days=6)
        window_df = df_sorted[(df_sorted['推广日期'] >= start) & (df_sorted['推广日期'] <= end)]
        
        if not window_df.empty:
            for model, model_name in zip(models, model_names):
                reward, top_keywords = model.calculate_selected_reward(window_df)
                # print(f'模型：{model_name}, 推广日期：{start}, 推广计划名称：{name}, 选择关键词：{top_keywords}， 奖励：{reward}')
                results.append({
                    'Group': name,
                    'Model': model_name,
                    'Window_Start': start.date(),
                    'Window_End': end.date(),
                    'Reward': reward,
                    'Top_Keywords': top_keywords,
                    'Keyword_Count': len(top_keywords),
                    'Total_Impressions': window_df['曝光量'].sum()  # Sum of all impressions in the window
                })

results_df = pd.DataFrame(results)

# Calculate cumulative reward per model and date
cumulative_reward = results_df.groupby(['Model', 'Window_Start'])['Reward'].sum().reset_index()
cumulative_reward = cumulative_reward.sort_values('Window_Start')
cumulative_reward['Cumulative_Reward'] = cumulative_reward.groupby('Model')['Reward'].cumsum()
cumulative_reward.to_csv(f'/home/houwanlong/ad_project/agents/KPMAgent/data/re_{reward_type}_topk_{topk}_ks.csv')

# Plot cumulative returns for all models
plt.figure(figsize=(14, 7))
for model_name in model_names:
    model_data = cumulative_reward[cumulative_reward['Model'] == model_name]
    plt.plot(model_data['Window_Start'], model_data['Cumulative_Reward'], label=model_name)

plt.title(f'{reward_type}模型累计奖励对比', fontproperties=font)
plt.xlabel('日期', fontproperties=font)
plt.ylabel('累计奖励值', fontproperties=font)
# plt.yscale('log')  # Log scale for Y-axis
plt.legend(title='模型类型', prop=font)
plt.tight_layout()

save_dir = "/home/houwanlong/ad_project/agents/KPMAgent/plot"
os.makedirs(save_dir, exist_ok=True)  # Create directory if missing

save_path = os.path.join(save_dir, f"re_{reward_type}_topk_{topk}_ks.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight', format='png')

plt.show()
