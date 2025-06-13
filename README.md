# KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits

## Overview

KP-Agent is an innovative agentic system designed to enhance the efficiency and accuracy of keyword pruning in Sponsored Search Advertising (SSA). Unlike traditional approaches that focus on bid adjustment and keyword generation, KP-Agent addresses the often-overlooked yet critical task of keyword pruning—refining keyword sets to maximize campaign performance.

## Motivation

In SSA, advertisers frequently expand their keyword sets to reach broader audiences. However, an excessively large keyword set can dilute the advertising budget, reducing the effectiveness of high-value keywords. Regular pruning of low-value keywords is essential to concentrate resources on those with better returns, thereby improving overall campaign efficiency.

## Key Features

- **LLM Agentic System**: Utilizes a large language model (LLM) to autonomously reason and interact with dynamic advertising environments.
- **Domain Toolset**: Encodes SSA-specific domain knowledge to guide the pruning process.
- **Memory Module**: Provides few-shot examples and reflective feedback from past decisions to improve future performance.
- **Contextual Bandit Framework**: Models keyword pruning as a contextual bandit problem, enabling adaptive and data-driven decision-making.
- **Advertiser-Side Data**: Operates solely on advertiser-side data (e.g., KPIs), eliminating the need for proprietary user search query data.

## Results

Experiments on a dataset of 0.5 million SSA records from a pharmaceutical advertiser on Meituan (China's largest delivery platform) demonstrate that KP-Agent improves cumulative profit by up to **49.28%** over baseline methods. Ablation studies confirm the value of both domain knowledge and reflection mechanisms.

## Contributions

- Introduces KP-Agent, an LLM-powered agentic system for keyword pruning in SSA.
- Pioneers the use of advertiser-side data for keyword pruning, addressing both industrial and academic gaps.
- Demonstrates significant improvements in campaign performance through extensive experiments.