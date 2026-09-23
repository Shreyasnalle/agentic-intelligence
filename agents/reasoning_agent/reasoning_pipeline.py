import json
from pathlib import Path
from typing import Dict, Any, Union
from langchain_core.runnables import RunnableLambda

try:
    from agents.reasoning_agent.reason_agent import ReasoningAgent
except ModuleNotFoundError:
    from reason_agent import ReasoningAgent


# Loads reasoning Q&A pairs from JSON file path or dictionary
def load_reasoning_input(source: Union[str, Dict[str, str]]) -> Dict[str, str]:
    if isinstance(source, dict):
        return source.get("reasoning_agent", source)

    path = Path(source)
    if not path.exists():
        # Fallback to routed_tasks.json if source path not found
        alt_path = Path("routed_tasks.json")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Reasoning input file not found at: {source}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "reasoning_agent" in data and isinstance(data["reasoning_agent"], dict):
        return data["reasoning_agent"]

    return data


# Runs reasoning evaluation pipeline using LCEL runnable chain
def run_reasoning_pipeline(
    reasoning_qa_json: Union[str, Dict[str, str]] = "routed_tasks.json",
) -> Dict[str, Any]:
    reasoning_agent = ReasoningAgent()

    pipeline_chain = (
        RunnableLambda(load_reasoning_input)
        | RunnableLambda(lambda qa: {
            "routed_reasoning_questions": len(qa),
            "analysis": reasoning_agent.analyze_reasoning(qa),
        })
    )

    return pipeline_chain.invoke(reasoning_qa_json)


if __name__ == "__main__":
    result = run_reasoning_pipeline()
    print(json.dumps(result, indent=2, ensure_ascii=False))
