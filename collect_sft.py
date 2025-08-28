from dotenv import load_dotenv
load_dotenv()

import art
import asyncio
from typing import List
from rollout import rollout
from art.trajectories import get_messages
import os
from art.langgraph import wrap_rollout
from art.langgraph.llm_wrapper import create_messages_from_logs
from tqdm.asyncio import tqdm
from dataclasses import dataclass
import json
import polars as pl

@dataclass
class Scenario:
    prompt: str
    prompt_id: str

async def collect_training_data(
    model: art.Model,
    sft_scenarios: List[Scenario]
):
    sft_scenarios = sft_scenarios + sft_scenarios

    sft_trajectories = await tqdm.gather(
        *(wrap_rollout(model, rollout)(scenario.prompt, scenario.prompt_id) for scenario in sft_scenarios),
        desc=f"collecting {model.name}",
    )

    valid_trajectories = [t for t in sft_trajectories if isinstance(t, art.Trajectory)]

    training_data = []

    for traj in valid_trajectories:
        if traj.reward < 0.45:
            continue

        training_data.append({"messages": get_messages(traj.messages_and_choices), "tools": traj.tools})
        for histroy in traj.additional_histories:
            training_data.append({"messages": get_messages(histroy.messages_and_choices), "tools": histroy.tools})

    with open("training-data.jsonl", 'w') as f:
        for data in training_data:
            f.write(json.dumps(data) + '\n')
    
    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))
    print(avg_metrics)

    return avg_metrics


async def train(model: art.TrainableModel):
    model = art.Model(
        name="o3", project="o3-sft",
        inference_model_name="o3",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1/",
    )

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

    await collect_training_data(model, sft_scenarios)


if __name__ == "__main__":
    asyncio.run(train(None))

