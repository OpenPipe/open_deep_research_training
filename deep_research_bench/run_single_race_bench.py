from deep_research_bench.deepresearch_bench_race import CRITERIA_FILE, REFERENCE_FILE, process_single_item, AIClient, MAX_RETRIES
import json
import threading
from tqdm import tqdm
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

CRITERIA_FILE = os.path.join(MODULE_DIR, CRITERIA_FILE)
REFERENCE_FILE = os.path.join(MODULE_DIR, REFERENCE_FILE)
PROMPT_FILE = os.path.join(MODULE_DIR, 'data/prompt_data/query.jsonl')

def single_race_bench(prompt_id, report):
    with open(CRITERIA_FILE) as f:
        for line in f.readlines():
            critera = json.loads(line)
            if critera['id'] == prompt_id:
                break
        else:
            raise "No criteria for prompt"

    
    with open(REFERENCE_FILE) as f:
        for line in f.readlines():
            reference = json.loads(line)
            if reference['id'] == prompt_id:
                break
        else:
            raise "No reference for prompt"
    
    with open(PROMPT_FILE) as f:
        for line in f.readlines():
            prompt = json.loads(line)
            if prompt['id'] == prompt_id:
                break
        else:
            raise "No reference for prompt"
        
    llm_client = AIClient()
    
    lock = threading.Lock()

    print(prompt)
    print(prompt['prompt'])
    # print(critera, reference)
    with tqdm(total=1, desc=f"Scoring") as pbar:
        result = process_single_item(prompt, {prompt['prompt']: {'id': prompt_id, 'prompt': prompt['prompt'], 'article': report}}, {prompt['prompt']: reference}, {prompt['prompt']: critera}, llm_client, lock, pbar, MAX_RETRIES, prompt['language'])
    print(result)
    return result
