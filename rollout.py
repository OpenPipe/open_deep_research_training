from open_deep_research.deep_researcher import deep_researcher_builder
from langgraph.checkpoint.memory import MemorySaver
import uuid
import json
import asyncio

from deep_research_bench.run_single_race_bench import single_race_bench
from art import Trajectory

from dotenv import load_dotenv
load_dotenv()

max_structured_output_retries = 3
allow_clarification = False
max_concurrent_research_units = 10
search_api = "tavily" # NOTE: We use Tavily to stay consistent
max_researcher_iterations = 6
max_react_tool_calls = 10
summarization_model = "openai:gpt-4.1-mini"
summarization_model_max_tokens = 8192
research_model = "openai:gpt-4.1"
research_model_max_tokens = 10000
compression_model = "openai:gpt-4.1"
compression_model_max_tokens = 10000
final_report_model = "openai:gpt-4.1"
final_report_model_max_tokens = 10000

graph_semaphore = asyncio.Semaphore(8)

async def rollout(
    prompt: str,
    prompt_id: int,
):
    report = None

    async with graph_semaphore:
        graph = deep_researcher_builder.compile(checkpointer=MemorySaver())
        config = {
            "configurable": {
                "thread_id": str(uuid.uuid4()),
            }
        }
        # NOTE: Configure the right dataset and evaluators
        config["configurable"]["max_structured_output_retries"] = max_structured_output_retries
        config["configurable"]["allow_clarification"] = allow_clarification
        config["configurable"]["max_concurrent_research_units"] = max_concurrent_research_units
        config["configurable"]["search_api"] = search_api
        config["configurable"]["max_researcher_iterations"] = max_researcher_iterations
        config["configurable"]["max_react_tool_calls"] = max_react_tool_calls
        config["configurable"]["summarization_model"] = summarization_model
        config["configurable"]["summarization_model_max_tokens"] = summarization_model_max_tokens
        config["configurable"]["research_model"] = research_model
        config["configurable"]["research_model_max_tokens"] = research_model_max_tokens
        config["configurable"]["compression_model"] = compression_model
        config["configurable"]["compression_model_max_tokens"] = compression_model_max_tokens
        config["configurable"]["final_report_model"] = final_report_model
        config["configurable"]["final_report_model_max_tokens"] = final_report_model_max_tokens
        # NOTE: We do not use MCP tools to stay consistent
        try:
            final_state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config
            )
            report = final_state['messages'][1].content
        except Exception as e:
            print('failed totally')
            print(e)
    
    if report:
        try:
            score = single_race_bench(prompt_id, report)
            return Trajectory(messages_and_choices=[], reward=score['overall_score'])
        except Exception as e:
            print(e)
    
    return Trajectory(messages_and_choices=[], reward=0)
