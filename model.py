from pandas import DataFrame
import pandas as pd
import json

# selected by KP-Agent
# [Imports moved to top of file for PEP8 compliance]
import ast
from kp_agent.config import openai_config, llm_config_list
from kp_agent.toolset_high import *
from kp_agent.kp_agent import KPAgent
import autogen
from kp_agent.prompts_kp import KPAgent_3Shots_Knowledge
from autogen.coding import LocalCommandLineCodeExecutor
import os
import tempfile
from openai import OpenAI

def reward_func(df: DataFrame, top_keywords: list, type: str):
    '''
    Input: A dataframe that contains the rows of selected keywords; The type of reward.
    Output: reward
    '''

    if type=='PV':
        #选取最后一天的数据
        top_keywords_df = df[df['关键词'].isin(top_keywords)]
        last_date = top_keywords_df['推广日期'].max()
        df_last_date = df[df['推广日期'] == last_date]
        #计算平均PV,PV方差
        reward_mean = df[df['推广日期'] == last_date]['PV'].mean()
        reward_sd = df[df['推广日期'] == last_date]['PV'].std()
        #计算被选中关键词reward
        top_keywords_df = top_keywords_df[top_keywords_df['推广日期'] == last_date]
        reward = top_keywords_df['PV']
        #计算被选中关键词reward的z score
        reward_z = (reward-reward_mean)/reward_sd
        return reward_z.sum()
    
    elif type=='流量CPC效率比_PV':
        #选取最后一天的数据
        top_keywords_df = df[df['关键词'].isin(top_keywords)]
        last_date = top_keywords_df['推广日期'].max()
        df_last_date = df[df['推广日期'] == last_date]
        #计算平均流量CPC效率比_PV,流量CPC效率比_PV方差
        reward_mean = df[df['推广日期'] == last_date]['流量CPC效率比_PV'].mean()
        reward_sd = df[df['推广日期'] == last_date]['流量CPC效率比_PV'].std()
        #计算被选中关键词reward
        top_keywords_df = top_keywords_df[top_keywords_df['推广日期'] == last_date]
        reward = top_keywords_df['流量CPC效率比_PV']
        #计算被选中关键词reward的z score
        reward_z = (reward-reward_mean)/reward_sd
        return reward_z.sum()

    elif type=='利润':
        #选取最后一天的数据
        top_keywords_df = df[df['关键词'].isin(top_keywords)]
        last_date = top_keywords_df['推广日期'].max()
        df_last_date = df[df['推广日期'] == last_date]
        #计算平均流量CPC效率比_PV,流量CPC效率比_PV方差
        reward_mean = df[df['推广日期'] == last_date]['利润_s1'].mean()
        reward_sd = df[df['推广日期'] == last_date]['利润_s1'].std()
        #计算被选中关键词reward
        top_keywords_df = top_keywords_df[top_keywords_df['推广日期'] == last_date]
        reward = top_keywords_df['利润_s1']
        #计算被选中关键词reward的z score
        reward_z = (reward-reward_mean)/reward_sd
        return reward.sum()
    return

class Model():

    def __init__(self, topk, reward_type='PV'):
        self.topk = topk
        self.reward_type = reward_type

    def select_word(self, df: DataFrame) -> list:
        top_keywords = DataFrame()
        return top_keywords['关键词'].tolist()

    def calculate_selected_reward(self, df: DataFrame) -> float:
        # Ensure the date is in datetime format
        df['推广日期'] = pd.to_datetime(df['推广日期'])
        
        # Get the topk keywords
        top_keywords = self.select_word(df)
        
        # Calculate reward using reward_func with the specified type
        reward = reward_func(df, top_keywords, self.reward_type)
        
        return reward, top_keywords

#Maximum
class Model_Benchmark(Model):

    def select_word(self, df: DataFrame) -> list:
        '''
        Input: A dataframe with columns '推广日期','推广计划名称','关键词',
                                        '利润','曝光量','点击量','点击率(*100%)',
                                        '转化率(*100%)','点击单价(单位元)','原价GTV(单位元)', 'label'
        Output: The topk keywords(关键词)
        '''
        # Group by promotion plan and keyword, then sum the labels
        keyword_last = df.groupby('关键词')[self.reward_type].last().reset_index() # 流量CPC效率比_PV
        
        # Sort by label sum in descending order and select topk keywords
        top_keywords = keyword_last.sort_values(self.reward_type, ascending=False).head(self.topk)
        
        return top_keywords['关键词'].tolist()

#Select word by impression
class Model_Impression(Model):

    def select_word(self, df: DataFrame) -> list:
        '''
        Input: A dataframe with columns '推广日期','推广计划名称','关键词',
                                        '利润','曝光量','点击量','点击率(*100%)',
                                        '转化率(*100%)','点击单价(单位元)','原价GTV(单位元)', 'label'
        Output: The topk keywords(关键词)
        '''
        # Group by promotion plan and keyword, then sum the labels
        keyword_sums = df.groupby('关键词')['曝光量'].sum().reset_index()
        
        # Sort by label sum in descending order and select topk keywords
        top_keywords = keyword_sums.sort_values('曝光量', ascending=False).head(self.topk)
        
        return top_keywords['关键词'].tolist()

#Select word by click
class Model_Click(Model):

    def select_word(self, df: DataFrame) -> list:
        '''
        Input: A dataframe with columns '推广日期','推广计划名称','关键词',
                                        '利润','曝光量','点击量','点击率(*100%)',
                                        '转化率(*100%)','点击单价(单位元)','原价GTV(单位元)', 'label'
        Output: The topk keywords(关键词)
        '''
        # Group by promotion plan and keyword, then sum the labels
        keyword_sums = df.groupby('关键词')['点击量'].sum().reset_index()
        
        # Sort by label sum in descending order and select topk keywords
        top_keywords = keyword_sums.sort_values('点击量', ascending=False).head(self.topk)
        
        return top_keywords['关键词'].tolist()

#Select word by ctr
class Model_CTR(Model):

    def select_word(self, df: DataFrame) -> list:
        '''
        Input: A dataframe with columns '推广日期','推广计划名称','关键词',
                                        '利润','曝光量','点击量','点击率(*100%)',
                                        '转化率(*100%)','点击单价(单位元)','原价GTV(单位元)', 'label'
        Output: The topk keywords(关键词)
        '''
        # Group by promotion plan and keyword, then sum the labels
        keyword_sums = df.groupby('关键词')['点击率(*100%)'].sum().reset_index()
        
        # Sort by label sum in descending order and select topk keywords
        top_keywords = keyword_sums.sort_values('点击率(*100%)', ascending=False).head(self.topk)
        
        return top_keywords['关键词'].tolist()

#Select word by ctr
class Model_CVR(Model):

    def select_word(self, df: DataFrame) -> list:
        '''
        Input: A dataframe with columns '推广日期','推广计划名称','关键词',
                                        '利润','曝光量','点击量','点击率(*100%)',
                                        '转化率(*100%)','点击单价(单位元)','原价GTV(单位元)', 'label'
        Output: The topk keywords(关键词)
        '''
        # Group by promotion plan and keyword, then sum the labels
        keyword_sums = df.groupby('关键词')['转化率(*100%)'].sum().reset_index()
        
        # Sort by label sum in descending order and select topk keywords
        top_keywords = keyword_sums.sort_values('转化率(*100%)', ascending=False).head(self.topk)
        
        return top_keywords['关键词'].tolist()

# selected by impression trend
class Model_TrendImpression(Model):

    def select_word(self, df: DataFrame) -> list:
        df['推广日期'] = pd.to_datetime(df['推广日期'])
        keyword_slopes = {}

        for keyword in df['关键词'].unique():
            keyword_df = df[df['关键词'] == keyword]
            grouped = keyword_df.groupby('推广日期', as_index=False)['曝光量'].sum()
            grouped = grouped.sort_values('推广日期')

            # if len(grouped) < 2:
            #     continue  # Skip keywords with insufficient data points

            min_date = grouped['推广日期'].min()
            grouped['days'] = (grouped['推广日期'] - min_date).dt.days
            x = grouped['days'].to_numpy()
            y = grouped['曝光量'].to_numpy()

            # Calculate slope using linear regression formula
            n = len(x)
            mean_x = x.mean()
            mean_y = y.mean()
            covariance = ((x - mean_x) * (y - mean_y)).sum()
            variance_x = ((x - mean_x) ** 2).sum()

            if variance_x == 0:
                slope = 0
            else:
                slope = covariance / variance_x

            keyword_slopes[keyword] = slope

        slopes_df = pd.DataFrame({
            '关键词': list(keyword_slopes.keys()),
            'slope': list(keyword_slopes.values())
        })
        # slopes_df = slopes_df[slopes_df['slope'] > 0]
        slopes_df = slopes_df.sort_values('slope', ascending=False)
        top_keywords = slopes_df.head(self.topk)

        return top_keywords['关键词'].tolist()

class Model_KPAgent(Model):

    def __init__(self, topk, reward_type='PV', llm = 'gpt-4o-mini-2024-07-18', seed = 42):
        self.topk = topk
        self.reward_type = reward_type
        self.config_list = [openai_config(llm)]
        self.llm_config = llm_config_list(seed, self.config_list)

        #initialize code writer
        self.chatbot = autogen.agentchat.AssistantAgent(
            name="chatbot",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
            llm_config=self.llm_config,
        )
        
        #initialize reflection generator
        self.refl_llm = OpenAI(
            api_key=self.config_list[0]["api_key"],
            base_url=self.config_list[0]["base_url"],
            # api_version=config["api_version"],
        )

        #initialize memory
        self.long_term_memory = []
        init_memory = KPMAgent_3Shots_Knowledge
        init_memory = init_memory.split('\n\n')
        for i in range(len(init_memory)):
            item = init_memory[i]
            item = item.split('Overview:')[-1]
            overview = item.split('\nKnowledge:\n')[0]
            item = item.split('\nKnowledge:\n')[-1]
            knowledge = item.split('\nReward:')[0]
            item = item.split('\nReward:\n')[-1]
            reward = item.split('\nSolution:')[0]
            code = item.split('\nSolution:')[-1]
            new_item = {"overview": overview, "knowledge": knowledge, "reward":reward, "code": code}
            self.long_term_memory.append(new_item)

        #initialize code excecutor
        temp_dir = tempfile.TemporaryDirectory() # Create a temporary directory to store the code files.
        self.executor = LocalCommandLineCodeExecutor( # Create a local command line code executor.
            timeout=10,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        #define kpagent
        self.user_proxy = KPAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"executor": self.executor},
            config_list=self.config_list,
        )

        #register function for kpagent
        self.user_proxy.register_function(
            function_map={
                "python": run_code
            }
        )

    def get_overview(self, window_df: DataFrame):
        number_of_keyword = len(window_df['keyword'].unique())
    
        # Impression stats
        impression_max = window_df['impression'].max()
        impression_75 = window_df['impression'].quantile(0.75)
        impression_50 = window_df['impression'].median()
        impression_25 = window_df['impression'].quantile(0.25)
        impression_min = window_df['impression'].min()
    
        # CTR stats (assuming ctr_*100 means percentage values)
        ctr_max = window_df['ctr'].max()
        ctr_75 = window_df['ctr'].quantile(0.75)
        ctr_50 = window_df['ctr'].median()
        ctr_25 = window_df['ctr'].quantile(0.25)
        ctr_min = window_df['ctr'].min()
    
        # CVR stats (assuming cvr_*100 means percentage values)
        cvr_max = window_df['cvr'].max()
        cvr_75 = window_df['cvr'].quantile(0.75)
        cvr_50 = window_df['cvr'].median()
        cvr_25 = window_df['cvr'].quantile(0.25)
        cvr_min = window_df['cvr'].min()
    
        # Profit stats
        profit_max = window_df['profit'].max()
        profit_75 = window_df['profit'].quantile(0.75)
        profit_50 = window_df['profit'].median()
        profit_25 = window_df['profit'].quantile(0.25)
        profit_min = window_df['profit'].min()

        overview_template = f'''
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: {number_of_keyword}
- Impression stats: Max={impression_max:.2f}, 75%={impression_75:.2f}, Median={impression_50:.2f}, 25%={impression_25:.2f}, Min={impression_min:.2f}
- CTR stats (%): Max={ctr_max:.2f}, 75%={ctr_75:.2f}, Median={ctr_50:.2f}, 25%={ctr_25:.2f}, Min={ctr_min:.2f}
- CVR stats (%): Max={cvr_max:.2f}, 75%={cvr_75:.2f}, Median={cvr_50:.2f}, 25%={cvr_25:.2f}, Min={cvr_min:.2f}
- Profit stats: Max={profit_max:.2f}, 75%={profit_75:.2f}, Median={profit_50:.2f}, 25%={profit_25:.2f}, Min={profit_min:.2f}
'''
    
        return overview_template

    def select_word(self, window_df: DataFrame) -> list:
        window_df['推广日期'] = pd.to_datetime(window_df['推广日期'])
        window_df_copy = window_df.copy()
        #rename column for processing
        window_df = window_df.rename(columns={
            '推广日期': 'date',
            '推广计划名称': 'campaign_name',
            '关键词': 'keyword',
            '利润': 'profit',
            '曝光量': 'impression',
            '点击量': 'click',
            '点击率(*100%)': 'ctr',  # Explicitly indicating percentage multiplied by 100
            '转化率(*100%)': 'cvr',  # Explicitly indicating percentage multiplied by 100
            '点击单价(单位元)': 'cpc',
            '原价GTV(单位元)': 'gtv',
            'PV': 'present_value',
            '当日效率比': 'daily_efficiency_ratio',
            '流量CPC效率比_PV': 'traffic_cpc_efficiency_ratio_pv'
        })
        output_dir = "kp-agent/kp_agent/data"
        window_df.to_csv(os.path.join(output_dir, f"data.csv"), index=False)
        #constuct overview for each input dataframe
        window_overview = self.get_overview(window_df)
        #update kpmagent's memory
        self.user_proxy.update_memory(3, self.long_term_memory) # first argument is the number of shot
        for attempt in range(20):
            try:
                # Start chat attempt
                self.user_proxy.initiate_chat(
                    self.chatbot,
                    message=window_overview,
                )
                # If successful, break out of the loop
                print(f"Chat initiated successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 19:  # Only wait if we're going to try again
                    print("Waiting 10 seconds before retrying...")
                    time.sleep(10)
            # If we exit the loop naturally, all attempts failed
                else:
                    print("Max retry attempts reached (20). Giving up.")
                    return []
        logs = self.user_proxy._oai_messages
        logs_string = []
        logs_string.append(str(window_overview))
        for agent in list(logs.keys()):
            for j in range(len(logs[agent])):
                if logs[agent][j]['content'] != None:
                    logs_string.append(logs[agent][j]['content'])
                else:
                    argums = logs[agent][j]['function_call']['arguments']
                    if type(argums) == dict and 'cell' in argums.keys():
                        logs_string.append(argums['cell'])
                    else:
                        logs_string.append(argums)

        #the keyword list
        keywords = logs_string[-2]
        if keywords!='':
            keywords = ast.literal_eval(keywords)
        else:
            keywords = []

        #calculate reward for RL
        # self.user_proxy.reward = self.calculate_reward(keywords[:self.topk], window_df_copy)
        reward = self.calculate_reward(keywords[:self.topk], window_df_copy)
        #generate reflection
        query_message = f"Overview: {window_overview}\nKnowledge: {self.user_proxy.knowledge}\nReward: {reward}\nCode: {self.user_proxy.code}\n\nPlease generate a reflection based on the above information."
        messages = [{"role":"system","content":"You are an AI assistant that helps for reflection generation, you will be provided with the overview of the input dataframe, the knowledge you have learned, the reward you have received, and the code you have generated. Please generate one or two sentence of reflection based on these information."},
                    {"role":"user","content": query_message}]
        self.user_proxy.reward = self.refl_llm.chat.completions.create(
                    model=self.config_list[0]["model"],
                    messages = messages,
                    temperature=0,
                    max_tokens=800,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None).choices[0].message.content.strip()




        logs_string = '\n----------------------------------------------------------\n'.join(logs_string)

        store_item = {
            'campaign_name':window_df['campaign_name'].unique()[0],
            'logs_string':logs_string,
            'date':window_df['date'].unique()[-1].strftime('%Y-%m-%d')
        } #append it to a file under "kp-agent/kp_agent/data"

        file_path = f"kp-agent/kp_agent/data/llm_{self.llm_config['config_list'][0]['model']}_re_{self.reward_type}_topk_{self.topk}.json"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Append store_item to the file in JSON format
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(store_item, f, ensure_ascii=False) 
            f.write('\n')  # Add a newline to separate entries  

        if '"cell": "' in logs_string:
            last_code_start = logs_string.rfind('"cell": "')
            last_code_end = logs_string.rfind('"\n}')
            last_code = logs_string[last_code_start+9:last_code_end]
        else:
            last_code_end = logs_string.rfind('Solution:')
        prediction_end = logs_string.rfind('TERMINATE')
        prediction = logs_string[last_code_end:prediction_end]

        if (type(keywords)==list):
            new_item = {"overview": window_overview, "knowledge": self.user_proxy.knowledge, "reward": self.user_proxy.reward, "code": self.user_proxy.code}
            self.long_term_memory.append(new_item)


        return keywords[:self.topk]
    
    def calculate_reward(self, keywords:list, df: DataFrame) -> float:
        # Ensure the date is in datetime format
        df['推广日期'] = pd.to_datetime(df['推广日期'])
        
        # Get the topk keywords
        top_keywords = keywords
        
        # Calculate reward using reward_func with the specified type
        reward = reward_func(df, top_keywords, self.reward_type)
        
        return reward

class Model_KPMAgent_ablation(Model):

    def __init__(self, topk, reward_type='PV', llm = 'gpt-4o-mini-2024-07-18', seed = 42):
        self.topk = topk
        self.reward_type = reward_type
        self.config_list = [openai_config(llm)]
        self.llm_config = llm_config_list(seed, self.config_list)

        #initialize code writer
        self.chatbot = autogen.agentchat.AssistantAgent(
            name="chatbot",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
            llm_config=self.llm_config,
        )
        
        #initialize reflection generator
        self.refl_llm = OpenAI(
            api_key=self.config_list[0]["api_key"],
            base_url=self.config_list[0]["base_url"],
            # api_version=config["api_version"],
        )

        #initialize memory
        self.long_term_memory = []
        init_memory = KPAgent_3Shots_Knowledge
        init_memory = init_memory.split('\n\n')
        for i in range(len(init_memory)):
            item = init_memory[i]
            item = item.split('Overview:')[-1]
            overview = item.split('\nKnowledge:\n')[0]
            item = item.split('\nKnowledge:\n')[-1]
            knowledge = item.split('\nReward:')[0]
            item = item.split('\nReward:\n')[-1]
            reward = item.split('\nSolution:')[0]
            code = item.split('\nSolution:')[-1]
            new_item = {"overview": overview, "knowledge": knowledge, "reward":reward, "code": code}
            self.long_term_memory.append(new_item)

        #initialize code excecutor
        temp_dir = tempfile.TemporaryDirectory() # Create a temporary directory to store the code files.
        self.executor = LocalCommandLineCodeExecutor( # Create a local command line code executor.
            timeout=10,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        #define kpagent
        self.user_proxy = KPAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"executor": self.executor},
            config_list=self.config_list,
        )

        #register function for kpagent
        self.user_proxy.register_function(
            function_map={
                "python": run_code
            }
        )

    def get_overview(self, window_df: DataFrame):
        number_of_keyword = len(window_df['keyword'].unique())
    
        # Impression stats
        impression_max = window_df['impression'].max()
        impression_75 = window_df['impression'].quantile(0.75)
        impression_50 = window_df['impression'].median()
        impression_25 = window_df['impression'].quantile(0.25)
        impression_min = window_df['impression'].min()
    
        # CTR stats (assuming ctr_*100 means percentage values)
        ctr_max = window_df['ctr'].max()
        ctr_75 = window_df['ctr'].quantile(0.75)
        ctr_50 = window_df['ctr'].median()
        ctr_25 = window_df['ctr'].quantile(0.25)
        ctr_min = window_df['ctr'].min()
    
        # CVR stats (assuming cvr_*100 means percentage values)
        cvr_max = window_df['cvr'].max()
        cvr_75 = window_df['cvr'].quantile(0.75)
        cvr_50 = window_df['cvr'].median()
        cvr_25 = window_df['cvr'].quantile(0.25)
        cvr_min = window_df['cvr'].min()
    
        # Profit stats
        profit_max = window_df['profit'].max()
        profit_75 = window_df['profit'].quantile(0.75)
        profit_50 = window_df['profit'].median()
        profit_25 = window_df['profit'].quantile(0.25)
        profit_min = window_df['profit'].min()

        overview_template = f'''
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: {number_of_keyword}
- Impression stats: Max={impression_max:.2f}, 75%={impression_75:.2f}, Median={impression_50:.2f}, 25%={impression_25:.2f}, Min={impression_min:.2f}
- CTR stats (%): Max={ctr_max:.2f}, 75%={ctr_75:.2f}, Median={ctr_50:.2f}, 25%={ctr_25:.2f}, Min={ctr_min:.2f}
- CVR stats (%): Max={cvr_max:.2f}, 75%={cvr_75:.2f}, Median={cvr_50:.2f}, 25%={cvr_25:.2f}, Min={cvr_min:.2f}
- Profit stats: Max={profit_max:.2f}, 75%={profit_75:.2f}, Median={profit_50:.2f}, 25%={profit_25:.2f}, Min={profit_min:.2f}
'''
    
        return overview_template

    def select_word(self, window_df: DataFrame) -> list:
        window_df['推广日期'] = pd.to_datetime(window_df['推广日期'])
        window_df_copy = window_df.copy()
        #rename column for processing
        window_df = window_df.rename(columns={
            '推广日期': 'date',
            '推广计划名称': 'campaign_name',
            '关键词': 'keyword',
            '利润': 'profit',
            '曝光量': 'impression',
            '点击量': 'click',
            '点击率(*100%)': 'ctr',  # Explicitly indicating percentage multiplied by 100
            '转化率(*100%)': 'cvr',  # Explicitly indicating percentage multiplied by 100
            '点击单价(单位元)': 'cpc',
            '原价GTV(单位元)': 'gtv',
            'PV': 'present_value',
            '当日效率比': 'daily_efficiency_ratio',
            '流量CPC效率比_PV': 'traffic_cpc_efficiency_ratio_pv'
        })
        output_dir = "kp-agent/kp_agent/data"
        window_df.to_csv(os.path.join(output_dir, f"data.csv"), index=False)
        #constuct overview for each input dataframe
        window_overview = self.get_overview(window_df)
        #update kpmagent's memory
        self.user_proxy.update_memory(3, self.long_term_memory) # first argument is the number of shot
        for attempt in range(20):
            try:
                # Start chat attempt
                self.user_proxy.initiate_chat(
                    self.chatbot,
                    message=window_overview,
                )
                # If successful, break out of the loop
                print(f"Chat initiated successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 19:  # Only wait if we're going to try again
                    print("Waiting 10 seconds before retrying...")
                    time.sleep(10)
            # If we exit the loop naturally, all attempts failed
                else:
                    print("Max retry attempts reached (20). Giving up.")
        logs = self.user_proxy._oai_messages
        logs_string = []
        logs_string.append(str(window_overview))
        for agent in list(logs.keys()):
            for j in range(len(logs[agent])):
                if logs[agent][j]['content'] != None:
                    logs_string.append(logs[agent][j]['content'])
                else:
                    argums = logs[agent][j]['function_call']['arguments']
                    if type(argums) == dict and 'cell' in argums.keys():
                        logs_string.append(argums['cell'])
                    else:
                        logs_string.append(argums)

        #the keyword list
        keywords = logs_string[-2]
        if keywords!='':
            keywords = ast.literal_eval(keywords)
        else:
            keywords = []

        #calculate reward for RL
        self.user_proxy.reward = ''




        logs_string = '\n----------------------------------------------------------\n'.join(logs_string)

        store_item = {
            'campaign_name':window_df['campaign_name'].unique()[0],
            'logs_string':logs_string,
            'date':window_df['date'].unique()[-1].strftime('%Y-%m-%d')
        } #append it to a file under "kp-agent/kp_agent/data"

        file_path = f"kp-agent/kp_agent/data/llm_{self.llm_config['config_list'][0]['model']}_re_{self.reward_type}_topk_{self.topk}.json"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Append store_item to the file in JSON format
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(store_item, f, ensure_ascii=False) 
            f.write('\n')  # Add a newline to separate entries  

        if '"cell": "' in logs_string:
            last_code_start = logs_string.rfind('"cell": "')
            last_code_end = logs_string.rfind('"\n}')
            last_code = logs_string[last_code_start+9:last_code_end]
        else:
            last_code_end = logs_string.rfind('Solution:')
        prediction_end = logs_string.rfind('TERMINATE')
        prediction = logs_string[last_code_end:prediction_end]

        if (type(keywords)==list):
            new_item = {"overview": window_overview, "knowledge": self.user_proxy.knowledge, "reward": self.user_proxy.reward, "code": self.user_proxy.code}
            self.long_term_memory.append(new_item)


        return keywords[:self.topk]
    
    def calculate_reward(self, keywords:list, df: DataFrame) -> float:
        # Ensure the date is in datetime format
        df['推广日期'] = pd.to_datetime(df['推广日期'])
        
        # Get the topk keywords
        top_keywords = keywords
        
        # Calculate reward using reward_func with the specified type
        reward = reward_func(df, top_keywords, self.reward_type)
        
        return reward

class Model_KPAgent_ablation2(Model):

    def __init__(self, topk, reward_type='PV', llm = 'gpt-4o-mini-2024-07-18', seed = 42):
        self.topk = topk
        self.reward_type = reward_type
        self.config_list = [openai_config(llm)]
        self.llm_config = llm_config_list(seed, self.config_list)

        #initialize code writer
        self.chatbot = autogen.agentchat.AssistantAgent(
            name="chatbot",
            system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
            llm_config=self.llm_config,
        )
        
        #initialize reflection generator
        self.refl_llm = OpenAI(
            api_key=self.config_list[0]["api_key"],
            base_url=self.config_list[0]["base_url"],
            # api_version=config["api_version"],
        )

        #initialize memory
        self.long_term_memory = []
        init_memory = KPAgent_3Shots_Knowledge
        init_memory = init_memory.split('\n\n')
        for i in range(len(init_memory)):
            item = init_memory[i]
            item = item.split('Overview:')[-1]
            overview = item.split('\nKnowledge:\n')[0]
            item = item.split('\nKnowledge:\n')[-1]
            knowledge = item.split('\nReward:')[0]
            item = item.split('\nReward:\n')[-1]
            reward = item.split('\nSolution:')[0]
            code = item.split('\nSolution:')[-1]
            new_item = {"overview": overview, "knowledge": knowledge, "reward":reward, "code": code}
            self.long_term_memory.append(new_item)

        #initialize code excecutor
        temp_dir = tempfile.TemporaryDirectory() # Create a temporary directory to store the code files.
        self.executor = LocalCommandLineCodeExecutor( # Create a local command line code executor.
            timeout=10,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        #define kpagent
        self.user_proxy = KPAgent(
            name="user_proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"executor": self.executor},
            config_list=self.config_list,
        )

        #register function for kpagent
        self.user_proxy.register_function(
            function_map={
                "python": run_code
            }
        )

    def get_overview(self, window_df: DataFrame):
        number_of_keyword = len(window_df['keyword'].unique())
    
        # Impression stats
        impression_max = window_df['impression'].max()
        impression_75 = window_df['impression'].quantile(0.75)
        impression_50 = window_df['impression'].median()
        impression_25 = window_df['impression'].quantile(0.25)
        impression_min = window_df['impression'].min()
    
        # CTR stats (assuming ctr_*100 means percentage values)
        ctr_max = window_df['ctr'].max()
        ctr_75 = window_df['ctr'].quantile(0.75)
        ctr_50 = window_df['ctr'].median()
        ctr_25 = window_df['ctr'].quantile(0.25)
        ctr_min = window_df['ctr'].min()
    
        # CVR stats (assuming cvr_*100 means percentage values)
        cvr_max = window_df['cvr'].max()
        cvr_75 = window_df['cvr'].quantile(0.75)
        cvr_50 = window_df['cvr'].median()
        cvr_25 = window_df['cvr'].quantile(0.25)
        cvr_min = window_df['cvr'].min()
    
        # Profit stats
        profit_max = window_df['profit'].max()
        profit_75 = window_df['profit'].quantile(0.75)
        profit_50 = window_df['profit'].median()
        profit_25 = window_df['profit'].quantile(0.25)
        profit_min = window_df['profit'].min()

        overview_template = f'''
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: {number_of_keyword}
- Impression stats: Max={impression_max:.2f}, 75%={impression_75:.2f}, Median={impression_50:.2f}, 25%={impression_25:.2f}, Min={impression_min:.2f}
- CTR stats (%): Max={ctr_max:.2f}, 75%={ctr_75:.2f}, Median={ctr_50:.2f}, 25%={ctr_25:.2f}, Min={ctr_min:.2f}
- CVR stats (%): Max={cvr_max:.2f}, 75%={cvr_75:.2f}, Median={cvr_50:.2f}, 25%={cvr_25:.2f}, Min={cvr_min:.2f}
- Profit stats: Max={profit_max:.2f}, 75%={profit_75:.2f}, Median={profit_50:.2f}, 25%={profit_25:.2f}, Min={profit_min:.2f}
'''
    
        return overview_template

    def select_word(self, window_df: DataFrame) -> list:
        window_df['推广日期'] = pd.to_datetime(window_df['推广日期'])
        window_df_copy = window_df.copy()
        #rename column for processing
        window_df = window_df.rename(columns={
            '推广日期': 'date',
            '推广计划名称': 'campaign_name',
            '关键词': 'keyword',
            '利润': 'profit',
            '曝光量': 'impression',
            '点击量': 'click',
            '点击率(*100%)': 'ctr',  # Explicitly indicating percentage multiplied by 100
            '转化率(*100%)': 'cvr',  # Explicitly indicating percentage multiplied by 100
            '点击单价(单位元)': 'cpc',
            '原价GTV(单位元)': 'gtv',
            'PV': 'present_value',
            '当日效率比': 'daily_efficiency_ratio',
            '流量CPC效率比_PV': 'traffic_cpc_efficiency_ratio_pv'
        })
        output_dir = "/home/houwanlong/ad_project/agents/KPMAgent/data"
        window_df.to_csv(os.path.join(output_dir, f"data.csv"), index=False)
        #constuct overview for each input dataframe
        window_overview = self.get_overview(window_df)
        #update kpmagent's memory
        self.user_proxy.update_memory(3, self.long_term_memory) # first argument is the number of shot
        for attempt in range(20):
            try:
                # Start chat attempt
                self.user_proxy.initiate_chat(
                    self.chatbot,
                    message=window_overview,
                )
                # If successful, break out of the loop
                print(f"Chat initiated successfully on attempt {attempt + 1}")
                break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 19:  # Only wait if we're going to try again
                    print("Waiting 10 seconds before retrying...")
                    time.sleep(10)
            # If we exit the loop naturally, all attempts failed
                else:
                    print("Max retry attempts reached (20). Giving up.")
        logs = self.user_proxy._oai_messages
        logs_string = []
        logs_string.append(str(window_overview))
        for agent in list(logs.keys()):
            for j in range(len(logs[agent])):
                if logs[agent][j]['content'] != None:
                    logs_string.append(logs[agent][j]['content'])
                else:
                    argums = logs[agent][j]['function_call']['arguments']
                    if type(argums) == dict and 'cell' in argums.keys():
                        logs_string.append(argums['cell'])
                    else:
                        logs_string.append(argums)

        #the keyword list
        keywords = logs_string[-2]
        if keywords!='':
            keywords = ast.literal_eval(keywords)
        else:
            keywords = []

        #calculate reward for RL
        self.user_proxy.reward = ''




        logs_string = '\n----------------------------------------------------------\n'.join(logs_string)

        store_item = {
            'campaign_name':window_df['campaign_name'].unique()[0],
            'logs_string':logs_string,
            'date':window_df['date'].unique()[-1].strftime('%Y-%m-%d')
        } #append it to a file under "/home/houwanlong/ad_project/agents/KPMAgent/data"

        file_path = f"kp-agent/kp_agent/data/llm_{self.llm_config['config_list'][0]['model']}_re_{self.reward_type}_topk_{self.topk}.json"

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Append store_item to the file in JSON format
        with open(file_path, 'a', encoding='utf-8') as f:
            json.dump(store_item, f, ensure_ascii=False) 
            f.write('\n')  # Add a newline to separate entries  

        if '"cell": "' in logs_string:
            last_code_start = logs_string.rfind('"cell": "')
            last_code_end = logs_string.rfind('"\n}')
            last_code = logs_string[last_code_start+9:last_code_end]
        else:
            last_code_end = logs_string.rfind('Solution:')
        prediction_end = logs_string.rfind('TERMINATE')
        prediction = logs_string[last_code_end:prediction_end]

        if (type(keywords)==list):
            new_item = {"overview": window_overview, "knowledge": '', "reward": self.user_proxy.reward, "code": self.user_proxy.code}
            self.long_term_memory.append(new_item)


        return keywords[:self.topk]
    
    def calculate_reward(self, keywords:list, df: DataFrame) -> float:
        # Ensure the date is in datetime format
        df['推广日期'] = pd.to_datetime(df['推广日期'])
        
        # Get the topk keywords
        top_keywords = keywords
        
        # Calculate reward using reward_func with the specified type
        reward = reward_func(df, top_keywords, self.reward_type)
        
        return reward


