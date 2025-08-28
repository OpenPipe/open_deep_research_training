<h1 align="center">DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents</h1>

<div align="center">
<a href="https://github.com/Ayanami0730/deep_research_bench/blob/main/LICENSE"><img src="https://img.shields.io/badge/Code_License-MIT-blue" alt="license"></a>
<a href="https://deepresearch-bench.github.io/"><img src="https://img.shields.io/badge/Website-DeepResearch-green" alt="website"></a>
<a href="https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?color=8A2BE2"></a>
<a href="https://arxiv.org/abs/2506.11763" target="_blank"><img src=https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv></a>
<a href="https://agi-eval.cn/evaluation/detail?id=67"><img src="https://img.shields.io/badge/🤝%20AGI--Eval-purple?color=8569f6" alt="AGI-Eval"></a>
</div>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

![Deep Research Agents Comparison Results](pics/leaderboard_0803.png)

# ✨ News
+ [3 Aug 2025] 🚀 We reproduced and evaluated [LangChain-Open-Deep-Research](https://github.com/langchain-ai/open_deep_research) (with GPT-4.1 + Tavily) as the first open-source framework evaluated on DeepResearch Bench, achieving 6th place among all deep research agents. This evaluation was conducted in collaboration with LangChain partners. Additionally, we partnered with [Nvidia-AIQ-Research](https://github.com/NVIDIA-AI-Blueprints/aiq-research-assistant) to evaluate their deep research solution. Updated results with new leaderboard visualization are now available. All detailed rankings and raw data are synchronized on our [Hugging Face Leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard). 
  
  **If you want to evaluate your deep research agent** Contact us at dumingxuan@mail.ustc.edu.cn to get official leaderboard ranking on DeepResearch Bench.
+ [18 July 2025] 🎉 We have established a partnership with **AGI-Eval** platform. DeepResearch Bench is now available on [**AGI-Eval**](https://agi-eval.cn/evaluation/detail?id=67), providing a more convenient evaluation interface for researchers and practitioners to test their deep research agents.
+ [15 July 2025] ⚡️⚡️ **Major Update**: Added comprehensive evaluation of **Kimi-Researcher**, **Doubao-DeepResearch**, and **Claude-Researcher**. Upgraded evaluation infrastructure with **Gemini-2.5-Pro** for RACE and **Gemini-2.5-Flash** for FACT evaluation. All raw research articles and evaluation scores are now available on our [**Hugging Face Leaderboard**](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) for comprehensive analysis and comparison.

For detailed evaluation results and comprehensive comparisons, please refer to the evaluation results table below.




## 📖 Overview

DeepResearch Bench addresses the absence of a comprehensive benchmark for systematically evaluating Deep Research Agents (DRAs). Our benchmark consists of **100 PhD-level research tasks**, each meticulously crafted by domain experts across **22 distinct fields**, including:

* 🔬 **Science & Technology**: Physics, chemistry, biology, environmental science, and engineering
* 💼 **Finance & Business**: investments, personal finance, marketing, and human resources
* 💻 **Software**: Topics related to the use of software and the internet
* 🌍 **Others**: Art & Design, Entertainment, History, Industrial, Transportation, Travel, and more


## Benchmark Construction

### Topic Distribution Analysis

To ensure DeepResearch Bench reflects real-world research demands, we analyzed **96,147 anonymized user queries** from web search-enabled LLM interactions.These queries were classified into **22 topic domains** based on the WebOrganizer taxonomy, revealing the authentic distribution of human deep research needs across different fields.

### Expert Task Collection

Guided by real-world demand distribution, we invited **PhD-level experts and senior practitioners** (5+ years experience) to design challenging research tasks within their domains. Each submission underwent rigorous manual screening for:

- **Quality**: High research standards and complexity
- **Clarity**: Clear task definitions and requirements  
- **Authenticity**: Grounded in real research scenarios
- **Challenge Level**: Testing upper limits of DRA capabilities

This process yielded **100 high-quality benchmark tasks** (50 Chinese, 50 English) that maintain the same topical balance as observed in real-world usage.


## Evaluation Framework

![Framework Overview](pics/framework.png)

DeepResearch Bench introduces two complementary evaluation methodologies designed to comprehensively assess Deep Research Agents:

### 🎯 RACE (Reference-based Adaptive Criteria-driven Evaluation)

RACE evaluates **report generation quality** through a sophisticated multi-step process:

- **Dynamic Criteria Generation**: Automatically generates task-specific evaluation criteria across four key dimensions:
  - 📚 **Comprehensiveness**: Coverage breadth and depth of the research topic
  - 🔍 **Insight/Depth**: Quality of analysis and insight generation  
  - 📋 **Instruction-Following**: Adherence to specific task requirements
  - 📖 **Readability**: Clarity, organization, and presentation quality

- **Reference-Based Scoring**: Compares target reports against high-quality reference reports to ensure discriminative evaluation
- **Weighted Assessment**: Uses dynamic weights adapted to each task's specific requirements

### 🔗 FACT (Framework for Factual Abundance and Citation Trustworthiness)

FACT evaluates **information retrieval and grounding capabilities** through:

- **Statement-URL Extraction**: Automatically extracts factual claims and their cited sources from generated reports
- **Deduplication**: Removes redundant statement-URL pairs to focus on unique factual claims
- **Support Verification**: Uses web scraping and LLM judgment to verify whether cited sources actually support the claims
- **Citation Metrics**: Calculates:
  - **Citation Accuracy**: Percentage of correctly supported citations
  - **Effective Citations**: Average number of verifiably supported citations per task


## 📊 Evaluation Results

### Main Results

Our comprehensive evaluation reveals significant performance variations across different model architectures and approaches:

| **Model** | **RACE Overall** | **RACE Comp.** | **RACE Depth** | **RACE Inst.** | **RACE Read.** | **FACT C. Acc.** | **FACT E. Cit.** |
|-----------|------------------|----------------|----------------|----------------|----------------|------------------|------------------|
| ***Deep Research Agent*** |
| Gemini-2.5-Pro Deep Research | **48.92** | **48.45** | **48.30** | <u>49.29</u> | **49.77** | <u>78.30</u> | **165.34** |
| OpenAI Deep Research | <u>46.45</u> | <u>46.46</u> | <u>43.73</u> | **49.39** | <u>47.22</u> | 75.01 | 39.79 |
| Claude-Researcher | 45.00 | 45.34 | 42.79 | 47.58 | 44.66 | - | - |
| Kimi-Researcher | 44.64 | 44.96 | 41.97 | 47.14 | 45.59 | - | - |
| Doubao-DeepResearch | 44.34 | 44.84 | 40.56 | 47.95 | 44.69 | 52.86 | <u>52.62</u> |
| Perplexity-Research | 40.46 | 39.10 | 35.65 | 46.11 | 43.08 | **82.63** | 31.20 |
| Grok Deeper Search | 38.22 | 36.08 | 30.89 | 46.59 | 42.17 | 73.08 | 8.58 |
| ***LLM with Search Tools*** |
| Perplexity-Sonar-Reasoning-Pro | **37.76** | <u>34.96</u> | <u>31.65</u> | **44.93** | **42.42** | 45.19 | 9.39 |
| Perplexity-Sonar-Reasoning | <u>37.75</u> | 34.73 | **32.59** | <u>44.42</u> | <u>42.39</u> | 52.58 | 13.37 |
| Claude-3.7-Sonnet w/Search | 36.63 | **35.95** | 31.29 | 44.05 | 36.07 | 87.32 | **24.51** |
| Perplexity-Sonar-Pro | 36.19 | 33.92 | 29.69 | 43.39 | 41.07 | 79.72 | <u>16.75</u> |
| Gemini-2.5-Pro-Preview | 31.90 | 31.75 | 24.61 | 40.24 | 32.76 | - | - |
| GPT-4o-Search-Preview | 30.74 | 27.81 | 20.44 | 41.01 | 37.60 | 86.63 | 5.05 |
| Perplexity-Sonar | 30.64 | 27.14 | 21.62 | 40.70 | 37.46 | 76.41 | 10.68 |
| GPT-4.1 w/Search | 29.31 | 25.59 | 18.42 | 40.63 | 36.49 | <u>89.85</u> | 4.27 |
| Gemini-2.5-Flash-Preview | 29.19 | 28.97 | 21.62 | 37.80 | 29.97 | - | - |
| GPT-4o-Mini-Search-Preview | 27.62 | 24.24 | 16.62 | 38.59 | 35.27 | 81.69 | 4.62 |
| GPT-4.1-Mini w/Search | 26.62 | 22.86 | 15.39 | 38.18 | 34.49 | 84.54 | 4.10 |
| Claude-3.5-Sonnet w/Search | 23.95 | 21.28 | 16.20 | 32.41 | 29.87 | **94.06** | 9.35 |

**Key Findings:**
- **Gemini-2.5-Pro Deep Research** achieves the highest overall performance (48.92) with exceptional depth and comprehensiveness, leading in all RACE metrics
- **OpenAI Deep Research** and **Claude-Researcher** follow closely, securing second and third place respectively, demonstrating strong research capabilities
- **Kimi-Researcher** and **Doubao-DeepResearch** also show competitive performance.
- **Deep Research Agents** significantly outperform traditional LLMs with search tools across all evaluation dimensions.
- **Citation accuracy** varies substantially across models, with Perplexity-Research achieving the highest accuracy among Deep Research Agents
- **Effective citation count** shows Gemini-2.5-Pro leading with around 165 citations per task, demonstrating superior information gathering capabilities

**Note on FACT Evaluation**: Due to differences between Jina AI's web scraping capabilities and internal scraping systems used by various companies, some links may fail to be scraped or return different content through Jina AI. Therefore, FACT evaluation results should be interpreted with caution and are provided for reference purposes only.

---

## 🛠️ Installation and Usage

### Prerequisites

- Python 3.9+
- Gemini API key (for LLM evaluation)
- Jina API key (for web scraping in FACT evaluation)

### Setup

```bash
git clone https://github.com/your-username/deep_research_bench.git
cd deep_research_bench
pip install -r requirements.txt
```

### API Configuration

Set the required API keys as environment variables:

```bash
# Set Gemini API key for LLM evaluation
export GEMINI_API_KEY="your_gemini_api_key_here"

# Set Jina API key for web scraping
export JINA_API_KEY="your_jina_api_key_here"
```


## Project Structure

```
deep_research_bench/
├── data/
│   ├── criteria_data/      # Evaluation criteria data
│   ├── prompt_data/        
│   │   └── query.jsonl     # ← 100 benchmark queries for your agent
│   └── test_data/          
│       ├── cleaned_data/   # Cleaned article data
│       └── raw_data/       # ← Put your model outputs here (model_name.jsonl)
├── prompt/                 # Prompt templates
├── utils/                  # Utility functions
├── deepresearch_bench_race.py  # RACE evaluation script
├── run_benchmark.sh        # ← Add your model names here, then run
└── requirements.txt        # Dependencies
```

**Quick Start Flow:**
1. Use queries from `data/prompt_data/query.jsonl` → Run your Deep Research Agent
2. Save outputs to `data/test_data/raw_data/<model_name>.jsonl`
3. Add model name to `TARGET_MODELS` in `run_benchmark.sh`
4. Run: `bash run_benchmark.sh`

## Quick Start

### 1. Prepare Your Model Data

Run your Deep Research Agent on the benchmark queries and save outputs in the required format:

**Input**: Use queries from `data/prompt_data/query.jsonl` (100 benchmark tasks)

**Output**: Save results to `data/test_data/raw_data/<model_name>.jsonl`

**Required format** (each line should contain):
```json
{
    "id": "task_id", 
    "prompt": "original_query_text", 
    "article": "generated_research_article_with_citations"
}
```

### 2. Configure Models to Evaluate

Edit `run_benchmark.sh` and add your model name:
```bash
TARGET_MODELS=("your-model-name")
```

### 3. Run Evaluation

```bash
bash run_benchmark.sh
```

Results will be saved to:
- RACE evaluation: `results/race/<model_name>/race_result.txt`
- FACT evaluation: `results/fact/<model_name>/fact_result.txt`

### Custom LLM Integration

If you're not using the official Gemini API or want to use other LLMs for evaluation, modify the `AIClient` class in `utils/api.py` to implement your custom LLM interface.

## Acknowledgements

We would like to express our gratitude to the following contributors who helped us collect evaluation data. Since many models and agents do not provide public APIs, manual data collection was necessary, and we deeply appreciate their dedicated efforts:

**Xin Yang**, **Jie Yang**, **Yawen Li**, **Xinyu Ouyang**, **Jiaqi He**, **Gefan Zhang**, **Jinfu Liao**, **Qiuyue Chen**, **Yulin Wang**, and **Lina Wang**.

Their contributions were essential to the comprehensive evaluation presented in this benchmark.

## Citation

If you use DeepResearch Bench in your research, please cite our paper:

```bibtex
@article{du2025deepresearch,
  author    = {Mingxuan Du and Benfeng Xu and Chiwei Zhu and Xiaorui Wang and Zhendong Mao},
  title     = {DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents},
  journal   = {arXiv preprint},
  year      = {2025},
}
``` 