CodeHeader = """
import sys
import os
target_dir = 'kp-agent/kp_agent'
if target_dir not in sys.path:
    sys.path.append(target_dir)
from tools import tabtools, calculator
Calculate = calculator.WolframAlphaCalculator
top_keywords_impression_ctr = tabtools.top_keywords_impression_ctr
top_keywords_conversion = tabtools.top_keywords_conversion
top_keywords_profitability = tabtools.top_keywords_profitability
top_keywords_efficiency = tabtools.top_keywords_efficiency
import pandas as pd
data = pd.read_csv('/home/houwanlong/ad_project/agents/KPMAgent/data/data.csv')
"""

RetrKnowledge = """Read the following descriptions, generate the background knowledge as the context information that could be helpful for selecting appropriate keywords.
(1) Use top_keywords_impression_ctr if impression stats show high variability (e.g., large gap between max/median) and CTR median is moderate, prioritizing keywords balancing broad reach (impressions) and engagement (CTR).
(2) Use top_keywords_conversion if CVR stats are skewed (e.g., high max but low median) and impression 25th percentile exceeds the dataset's noise threshold, ensuring statistically reliable keywords drive conversions (CVR × GTV).
(3) Use top_keywords_profitability if profit stats reveal outliers (e.g., high max but negative min), isolating keywords where net profit (GTV - costs) is positive and significant.
(4) Use top_keywords_efficiency if CPC varies widely and GTV/impression ratios are uneven, prioritizing keywords maximizing return per Yuan spent (GTV / ad spend).

Overview: 
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: 16
- Impression stats: Max=427.00, 75%=17.00, Median=7.00, 25%=3.00, Min=1.00
- CTR stats (%): Max=66.67, 75%=13.86, Median=0.00, 25%=0.00, Min=0.00
- CVR stats (%): Max=100.00, 75%=41.03, Median=0.00, 25%=0.00, Min=0.00
- Profit stats: Max=1021.13, 75%=44.83, Median=0.00, 25%=0.00, Min=-0.52
Knowledge:
- Given the extreme impression skew (max=427 vs. median=7), near-zero median CTR/CVR (median=0%), and profit outliers (max=1021.13 vs. median=0), use top_keywords_profitability to prioritize high-profit keywords (avoiding loss-making ones) and top_keywords_impression_ctr to identify keywords with above-median impressions (≥7) and non-zero CTR (e.g., max=66.67%). Generate both ranked lists, then select overlapping or top-ranked keywords to balance profitability and visibility.

Overview:
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: 9
- Impression stats: Max=14127.00, 75%=1397.00, Median=959.00, 25%=569.00, Min=48.00
- CTR stats (%): Max=12.50, 75%=5.85, Median=4.22, 25%=3.29, Min=1.33
- CVR stats (%): Max=44.12, 75%=27.83, Median=21.95, 25%=16.67, Min=0.00
- Profit stats: Max=7113.85, 75%=392.06, Median=226.39, 25%=69.50, Min=-4.15
Knowledge: 
- Given the high impression range (max=14,127 vs. median=959), moderate CTR (median=4.22%), and reliable CVR (median=21.95%) with profit outliers (max=7,113.85 vs. median=226.39), use top_keywords_impression_ctr to prioritize keywords balancing broad reach (impressions ≥959) and engagement (CTR >4.22%), and top_keywords_conversion to leverage statistically significant CVR (median=21.95%) and GTV. 

Overview: {overview}
Knowledge:
"""

SYSTEM_PROMPT = """You are a helpful AI assistant. Solve tasks using your coding and language skills.
In the following cases, suggest python code (in a python coding block) or shell script (in a sh
coding block) for the user to execute.
1. When you need to collect info, use the code to output the info you need, for example, browse or
search the web, download/read a file, print the content of a webpage or a file, get the current
date/time. After sufficient info is printed and the task is ready to be solved based on your
language skill, you can solve the task by yourself.
2. When you need to perform some task with code, use the code to perform the task and output the
result. Finish the task smartly.
Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be
clear which step uses code, and which step uses your language skill.
When using code, you must indicate the script type in the code block. The user cannot provide any
other feedback or perform any other action beyond executing the code you suggest. The user can't
modify your code. So do not suggest incomplete code which requires users to modify. Don't use a
code block if it's not intended to be executed by the user.
If you want the user to save the code in a file before executing it, put # filename: <filename>
inside the code block as the first line. Don't include multiple code blocks in one response. Do not
ask users to copy and paste the result. Instead, use 'print' function for the output when relevant.
Check the execution result returned by the user.
If the result indicates there is an error, fix the error and output the code again. Suggest the
full code instead of partial code or code changes. If the error can't be fixed or if the task is
not solved even after the code is executed successfully, analyze the problem, revisit your
assumption, collect additional info you need, and think of a different approach to try.
When you find an answer, verify the answer carefully. Include verifiable evidence in your response
if possible.
Reply "TERMINATE" in the end when everything is done."""

KPAgent_Message_Prompt = """Assume you are a Keyword Portfolio Management Agent with access to all keywords' advertising performance data in a campaign in sponsored search advertising:
- date: YYYY-MM-DD
- keyword: str
- impression: int
- click: int
- ctr: float
- conversions: int
- cvr: int
- cpc: float
- gtv: float

Write a python code to give a ranked list of keywords given the description of the input table. You can use the following functions:
(1) top_keywords_impression_ctr(data), which ranks keywords by combining total impression and average CTR to balance visibility and engagement for brand-awareness campaigns. The output is a list of keywords.
(2) top_keywords_conversion(data, min_impression), which prioritizes keywords with statistically significant impression, optimizing for conversion rate (CVR) and total business value (GTV) in performance-focused campaigns. The output is a list of keywords.
(3) top_keywords_profitability(data), which evaluates keywords by estimating net profit (GTV minus advertising costs) for ROI-driven campaign optimization. The output is a list of keywords.
(4) top_keywords_efficiency(data), which identifies keywords with the highest gross transaction value (GTV) per advertising Yuan spent to maximize budget efficiency. The output is a list of keywords.
Use the variable 'answer' to store the answer of the code. Here are some examples, examples with higer reward yields better performance:
{examples}
(END OF EXAMPLES)
Knowledge:
{knowledge}
Overview: {overview}
Solution: """

DEFAULT_USER_PROXY_AGENT_DESCRIPTIONS = {
    "ALWAYS": "An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
    "TERMINATE": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
    "NEVER": "A user that can run Python code or input command line commands at a Linux terminal and report back the execution results.",
}

CodeDebugger = """Given a overview of data:
{overview}
The user have written code with the following functions:
(1) top_keywords_impression_ctr(data), which ranks keywords by combining total impression and average CTR to balance visibility and engagement for brand-awareness campaigns. The output is a list of keywords.
(2) top_keywords_conversion(data, min_impression), which prioritizes keywords with statistically significant impression, optimizing for conversion rate (CVR) and total business value (GTV) in performance-focused campaigns. The output is a list of keywords.
(3) top_keywords_profitability(data), which evaluates keywords by estimating net profit (GTV minus advertising costs) for ROI-driven campaign optimization. The output is a list of keywords.
(4) top_keywords_efficiency(data), which identifies keywords with the highest gross transaction value (GTV) per advertising Yuan spent to maximize budget efficiency. The output is a list of keywords.

The code is as follows:
{code}

The execution result is:
{error_info}

Please check the code and point out the most possible reason to the error.
"""

KPAgent_3Shots_Knowledge = """
Overview: 
From the dataframe of SSA campaign performance, we have observed that:
    - Number of unique keywords: 16
    - Impression stats: Max=427.00, 75%=17.00, Median=7.00, 25%=3.00, Min=1.00
    - CTR stats (%): Max=66.67, 75%=13.86, Median=0.00, 25%=0.00, Min=0.00
    - CVR stats (%): Max=100.00, 75%=41.03, Median=0.00, 25%=0.00, Min=0.00
    - Profit stats: Max=1021.13, 75%=44.83, Median=0.00, 25%=0.00, Min=-0.52
Knowledge:
- Given the extreme impression skew (max=427 vs. median=7), near-zero median CTR/CVR (median=0%), and profit outliers (max=1021.13 vs. median=0), use top_keywords_profitability to prioritize high-profit keywords (avoiding loss-making ones) and top_keywords_impression_ctr to identify keywords with above-median impressions (≥7) and non-zero CTR (e.g., max=66.67%). Generate both ranked lists, then select overlapping or top-ranked keywords to balance profitability and visibility.
Reward:
1
Solution:
# Generate prioritized lists
profit_ranked = top_keywords_profitability(data)  # Profit descending
impression_ctr_ranked = top_keywords_impression_ctr(data)        # Score descending
# Filter & combine:
# 1. Take profitable keywords (ignore negative/zero profit)
positive_profit = [kw for kw in profit_ranked if data[data['keyword'] == kw]['profit'].sum() > 0]
# 2. Take high-impression-CTR keywords (ignore CTR=0%)
non_zero_ctr = [kw for kw in impression_ctr_ranked 
                if data[data['keyword'] == kw]['ctr'].mean() > 0]
# Merge and deduplicate
combined = list(dict.fromkeys(positive_profit + non_zero_ctr))
answer = combined

Overview:
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: 9
- Impression stats: Max=14127.00, 75%=1397.00, Median=959.00, 25%=569.00, Min=48.00
- CTR stats (%): Max=12.50, 75%=5.85, Median=4.22, 25%=3.29, Min=1.33
- CVR stats (%): Max=44.12, 75%=27.83, Median=21.95, 25%=16.67, Min=0.00
- Profit stats: Max=7113.85, 75%=392.06, Median=226.39, 25%=69.50, Min=-4.15
Knowledge: 
- Given the high impression range (max=14,127 vs. median=959), moderate CTR (median=4.22%), and reliable CVR (median=21.95%) with profit outliers (max=7,113.85 vs. median=226.39), use top_keywords_impression_ctr to prioritize keywords balancing broad reach (impressions ≥959) and engagement (CTR >4.22%), and top_keywords_conversion to leverage statistically significant CVR (median=21.95%) and GTV. 
Reward:
1
Solution:
# Calculate 25th percentile impression for conversion threshold
keyword_imp_totals = data.groupby("keyword")["impression"].sum()
min_imp_threshold = keyword_imp_totals.quantile(0.25)  # 569 based on overview
# Generate ranked lists
impression_ctr_ranked = top_keywords_impression_ctr(data)  # List[str]
conversion_ranked = top_keywords_conversion(data, min_impression=min_imp_threshold)  # List[str]
top_impression_ctr = impression_ctr_ranked
top_conversion = conversion_ranked
# Merge and prioritize overlapping keywords
combined = []
seen = set()
for kw in top_impression_ctr + top_conversion:
    if kw not in seen:
        seen.add(kw)
        combined.append(kw)
answer = combined

Overview:
From the dataframe of SSA campaign performance, we have observed that:
- Number of unique keywords: 17
- Impression stats: Max=366.00, 75%=12.25, Median=5.50, 25%=2.75, Min=1.00
- CTR stats (%): Max=50.00, 75%=14.72, Median=0.00, 25%=0.00, Min=0.00
- CVR stats (%): Max=100.00, 75%=25.00, Median=0.00, 25%=0.00, Min=0.00
- Profit stats: Max=898.93, 75%=42.25, Median=0.00, 25%=0.00, Min=-0.30"
Knowledge:
- Given the extreme impression skew (max=366 vs. median=5.5), zero median CTR/CVR (median=0%), and profit outliers (max=898.93 vs. median=0), use top_keywords_profitability to prioritize keywords with positive net profit (excluding min=-0.30) and top_keywords_impression_ctr to target keywords with impressions ≥5.5 (median) and CTR >0% (max=50%).
Reward:
1
Solution:
median_impression = (
    data.groupby('keyword')['impression']  # Group by keyword
    .sum()                                  # Get total impression per keyword
    .median()                               # Calculate median of summed impression
)
# Generate prioritized lists
profit_ranked = top_keywords_profitability(data)
impression_ctr_ranked = top_keywords_impression_ctr(data)
# Filter keywords
profitable_kws = [
    kw for kw in profit_ranked
    if profit_ranked[kw] > 0  # Exclude negative/zero profit
]
high_engagement_kws = [
    kw for kw in impression_ctr_ranked 
    if (data[data['keyword'] == kw]['impression'].sum() >= median_impression  # Use dynamic threshold
    and data[data['keyword'] == kw]['ctr'].mean() > 0)  # Exclude CTR=0%
]
# Merge & deduplicate
combined = []
seen = set()
for kw in profitable_kws + high_engagement_kws:
    if kw not in seen:
        seen.add(kw)
        combined.append(kw)
answer = combined
"""

if __name__ == "__main__":

    import sys
    import os
    target_dir = 'kp-agent/kp_agent'
    if target_dir not in sys.path:
        sys.path.append(target_dir)
    from tools import tabtools, calculator
    Calculate = calculator.WolframAlphaCalculator
    top_keywords_impression_ctr = tabtools.top_keywords_impression_ctr
    top_keywords_conversion = tabtools.top_keywords_conversion
    top_keywords_profitability = tabtools.top_keywords_profitability
    top_keywords_efficiency = tabtools.top_keywords_efficiency
    import pandas as pd
    data = pd.read_csv('kp-agent/kp_agent/data/data.csv')
    median_impression = (
        data.groupby('keyword')['impression']  # Group by keyword
        .sum()                                  # Get total impression per keyword
        .median()                               # Calculate median of summed impression
    )
    # Generate prioritized lists
    profit_ranked = top_keywords_profitability(data)
    impression_ctr_ranked = top_keywords_impression_ctr(data)
    # Filter keywords
    profitable_kws = [
        kw for kw in profit_ranked.index.tolist() 
        if profit_ranked[kw] > 0  # Exclude negative/zero profit
    ]
    high_engagement_kws = [
        kw for kw in impression_ctr_ranked 
        if (data[data['keyword'] == kw]['impression'].sum() >= median_impression  # Use dynamic threshold
        and data[data['keyword'] == kw]['ctr'].mean() > 0)  # Exclude CTR=0%
    ]
    # Merge & deduplicate
    combined = []
    seen = set()
    for kw in profitable_kws + high_engagement_kws:
        if kw not in seen:
            seen.add(kw)
            combined.append(kw)
    answer = combined
    print(answer)