# Portions adapted from Unsloth Notebooks (https://github.com/unslothai/notebooks)
# Copyright (c) Unsloth contributors.
# License: GNU LGPL v3.0.

from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-14B-Instruct",
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = True,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,  # Best to choose alpha = rank or rank*2
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,   # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

import json
from datasets import Dataset as HFDataset
import copy

def clean_messages(messages):
    msgs = copy.deepcopy(messages)
    for m in msgs:
        # drop empty tool_calls arrays
        if m.get("tool_calls") == []:
            m.pop("tool_calls", None)
        # ensure content exists and is a string
        if "content" not in m or m.get("content") is None:
            m["content"] = ""
    return msgs

all_conversations = []
with open("training-data.jsonl", "r") as f:
    for line in f:
        all_conversations.append(json.loads(line))

dataset_list = []
for conversation in all_conversations:
    messages = clean_messages(conversation["messages"])  # <-- use the cleaner
    tools = conversation.get("tools", None)
    dataset_list.append(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
    )

full_dataset = HFDataset.from_dict({"text": dataset_list})
split_dataset = full_dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

print(dataset_list[0])

from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        warmup_steps = 5,
        num_train_epochs = 10,
        learning_rate = 2e-5, 
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

from unsloth_zoo.dataset_utils import train_on_responses_only
QWEN_INSTRUCTION_PART = "<|im_start|>user\n"
QWEN_RESPONSE_PART = "<|im_start|>assistant\n"
trainer.data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
trainer = train_on_responses_only(
    trainer,
    instruction_part=QWEN_INSTRUCTION_PART,
    response_part=QWEN_RESPONSE_PART,
)

trainer_stats = trainer.train()

model.save_pretrained("model")
tokenizer.save_pretrained("model")

import art
from art.local import LocalBackend
import os
model = art.TrainableModel(name="sft-deep-re", project="deep-re", base_model="./model")

backend = LocalBackend()
backend._experimental_push_to_s3(model, s3_bucket=os.environ["BACKUP_BUCKET"],)