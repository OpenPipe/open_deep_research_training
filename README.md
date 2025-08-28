This repo contains [Langchain's open deep research](https://github.com/langchain-ai/open_deep_research) and [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](https://github.com/Ayanami0730/deep_research_bench) as well as portions adapted from [Unsloth Notebooks](https://github.com/unslothai/notebooks) for SFT training.

Along with these repos it also contains several scripts to enable training (both SFT and RL) of your own model to power your deep research agent.

### Overview

You'll have to fill out your .env with your Tavily api key (used to power websearches), Openai api key (used to collect samples for SFT) and Google api key (used to run the benchmark). You'll also have to add your AWS credentials and a backup bucket to store the SFT model to start RL run, as well as to store checkpoints during your RL run.

Additionaly we reccomend adding your `WANDB_API_KEY` for observability. 

Then you'll want to run these scripts in this order

```
uv run collect_sft.py # Collect samples for your sft training run. ~1 Hour
uv run run_sft.py # Run your sft training run. ~1 Hour
uv run run_train.py # Run your rl training run. >1 Day
```

### Modifications

We modified the DeepResearch Bench repo to add a new `run_single_race_bench.py` which allows you to run a single benchmark at a time, needed for running RL runs.

We modified the Open Deep Research repo to change the search to use Tavily's advanced search answering to enable training models with smaller context windows.