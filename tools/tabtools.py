import pandas as pd
import jsonlines
import json
import re
import sqlite3
import sys
import Levenshtein

def top_keywords_impression_ctr(data):
    '''
    Performance Ranking (Impressions + Engagement):
        Use when optimizing for both visibility (impressions) and engagement (CTR). 
        This combines reach and ad relevance in one metric.
        Best for campaigns focused on brand awareness with some performance consideration.
    '''
    return (data.groupby('keyword')
            .agg({'impression':'sum','ctr':'mean'})
            .eval('score = impression * ctr')
            .sort_values('score', ascending=False)
            .index.tolist()
           )

def top_keywords_impression(data):
    '''
    Performance Ranking Impressions Only:
        Use when optimizing for visibility (impressions) alone.
        This prioritizes reach over engagement.
    '''
    return (data.groupby('keyword')
            .agg({'impression':'sum'})
            .eval('score = impression')
            .sort_values('score', ascending=False)
            .index.tolist()
           )

def top_keywords_click(data):
    '''
    Performance Ranking Clicks Only:
        Use when optimizing for clicks alone.
        This prioritizes engagement over visibility.
    '''
    return (data.groupby('keyword')
            .agg({'click':'sum'})
            .eval('score = click')
            .sort_values('score', ascending=False)
            .index.tolist()
           )

def top_keywords_conversion(data, min_impressions=1000):
    '''
    Conversion-Oriented Ranking:
        Ideal for performance campaigns focused on conversions.
        The minimum impression filter ensures statistical significance.
        Combines conversion rate (CVR) with total business value (GTV).
    '''
    return (data.groupby('keyword')
            .filter(lambda x: x['impression'].sum() > min_impressions)
            .groupby('keyword')
            .agg({'cvr':'mean', 'gtv':'sum'})
            .eval('score = cvr * gtv')
            .sort_values('score', ascending=False)
            .index.tolist()
           )

def top_keywords_profitability(data):
    '''
    Profitability-Focused Ranking:
        Use when optimizing for net profitability.
        Calculates estimated profit by subtracting total cost (impressions * CTR * CPC) from gross transaction value (GTV).
        Crucial for ROI-focused campaigns.
    '''
    return (data.groupby('keyword')
            .apply(lambda x: (x['gtv'].sum() - x['cpc'].mean() * x['impression'].sum() * x['ctr'].mean()))
            .sort_values(ascending=False)
            .index.tolist()
           )

def top_keywords_efficiency(data):
    '''
    Efficiency Ranking (Cost vs Results):
        Calculates GTV generated per Yuan spent. Crucial for budget-constrained campaigns.
        Prioritizes keywords that deliver maximum value per advertising Yuan.
    '''
    return (data.groupby('keyword')
            .apply(lambda x: x['gtv'].sum() / (x['cpc'].mean() * x['impression'].sum()))
            .sort_values(ascending=False)
            .index.tolist()
           )