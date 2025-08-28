from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage

# Monkeypatch the AIMessage init to auto-fix args
_original_init = AIMessage.__init__

def _patched_init(self, **kwargs):
    tool_calls = kwargs.get("tool_calls", [])
    for call in tool_calls:
        if isinstance(call.get("args"), str):
            try:
                print(call['args'])
                call["args"] = json.loads(call["args"])
            except json.JSONDecodeError:
                call["args"] = {}
    _original_init(self, **kwargs)

AIMessage.__init__ = _patched_init

import art
from art.local import LocalBackend
import asyncio
from typing import List
from rollout import rollout
import os
from art.langgraph import wrap_rollout
from tqdm.asyncio import tqdm
from dataclasses import dataclass
import json
import os
import polars as pl

@dataclass
class Scenario:
    prompt: str
    prompt_id: str

async def benchmark_model(
    model: art.Model,
    val_scenarios: List[Scenario]
):
    val_scenarios = val_scenarios + val_scenarios

    val_trajectories = await tqdm.gather(
        *(wrap_rollout(model, rollout)(scenario.prompt, scenario.prompt_id) for scenario in val_scenarios),
        desc=f"validation {model.name}",
    )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

    if model._backend is not None:
        await model.log(valid_trajectories)
    
    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))
    print(avg_metrics)

    return avg_metrics


async def benchmark(model: art.TrainableModel):
    with LocalBackend() as backend:
        await model.register(backend)

        train_scenarios: List[Scenario] = []
        val_scenarios: List[Scenario] = []
        sft_scenarios: List[Scenario] = []

        PROMPT_FILE = 'deep_research_bench/data/prompt_data/query.jsonl'
        with open(PROMPT_FILE) as f:
            for line in f.readlines():
                obj = json.loads(line)
                if (obj['id']-1) % 50 >= 45:
                    val_scenarios.append(Scenario(prompt=obj["prompt"], prompt_id=obj["id"]))
                elif (obj['id']-1) % 50 >= 20:
                    sft_scenarios.append(Scenario(prompt=obj["prompt"], prompt_id=obj["id"]))
                else:
                    train_scenarios.append(Scenario(prompt=obj["prompt"], prompt_id=obj["id"]))
        
        await benchmark_model(model, val_scenarios)


if __name__ == "__main__":
    asyncio.run(benchmark(model = art.Model(
        name="gpt-4.1", project="deep-re",
        inference_model_name="gpt-4.1",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1/",
    )))

    # asyncio.run(benchmark(model = art.Model(
    #     name="sonnet-4", project="deep-re",
    #     inference_model_name="claude-sonnet-4-20250514",
    #     inference_api_key=os.getenv("ANTHROPIC_API_KEY"),
    #     inference_base_url="https://api.anthropic.com/v1/",
    # )))