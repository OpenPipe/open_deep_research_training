# Open Deep Research Training

This repo demonstrates how to train an LLM using GRPO to exceed SOTA performance at deep research. Specifically, you will be using the [ART](https://github.com/OpenPipe/ART) library to specialize an agent for [Langchain's open deep research](https://github.com/langchain-ai/open_deep_research) framework, and will evaluate your agent's performance using [DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](https://github.com/Ayanami0730/deep_research_bench).

In addition to the GRPO training step, you will also run an initial SFT training run to improve the model's baseline performance.

## Getting Started

### 1. Install dependencies

If you haven't already, install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Then install the project dependencies by running `uv sync`.

### 2. Install SkyPilot/RunPod

We'll be using `LocalBackend` to manage the GPU that your model will be trained on. In order to provision a GPU for your training run, you'll need to have SkyPilot installed on your machine and provide it with the credentials to spin up machines on at least one infra provider.

We recommend using RunPod because of their ease of use, but any infra provider that SkyPilot [supports](https://docs.skypilot.co/en/latest/overview.html#bringing-your-infra) will work.

Follow RunPod's **Getting Started** guide [here](https://docs.runpod.io/integrations/skypilot/). You'll have to provide a credit card to use RunPod, but you'll only pay for the time your GPUs are running.

### 3. Set up optional environment variables found in `.env.example`.

Copy `.env.example` to `.env` at the root of the repository, and fill in the values for the environment variables. If you're unsure about any of the values, refer to [ENV_INSTRUCTIONS.md](ENV_INSTRUCTIONS.md).

### 4. Run the scripts

You'll want to run these scripts in this order:

```
uv run collect_sft.py # Collect samples for your sft training run. ~1 Hour
uv run run_sft.py # Run your sft training run. ~1 Hour
uv run run_train.py # Run your rl training run. >1 Day
```

### 5. Generate the benchmarks

Run the benchmark script in the evaluate folder with the models you want to benchmark
```
uv run evaluate/benchmark_model.py
```

Then run the display_benchmarks.ipynb to display the results.

### Modifications

We modified the DeepResearch Bench repo to add a new `run_single_race_bench.py` which allows you to run a single benchmark at a time, needed for running RL runs.

We modified the Open Deep Research repo to change the search to use Tavily's advanced search answering to enable training models with smaller context windows.
