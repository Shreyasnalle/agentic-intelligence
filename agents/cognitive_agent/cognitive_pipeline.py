import json
from pathlib import Path
from typing import Dict, Any, Union
from langchain_core.runnables import RunnableLambda

try:
    from agents.cognitive_agent.cognitive_agent import CognitiveAgent
except ModuleNotFoundError:
    from cognitive_agent import CognitiveAgent


# Loads cognitive Q&A pairs from JSON file path or dictionary
def load_cognitive_input(source: Union[str, Dict[str, str]]) -> Dict[str, str]:
    if isinstance(source, dict):
        return source.get("cognitive_agent", source)

    path = Path(source)
    if not path.exists():
        alt_path = Path("routed_tasks.json")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Cognitive input file not found at: {source}")

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and "cognitive_agent" in data and isinstance(data["cognitive_agent"], dict):
        return data["cognitive_agent"]

    return data


# Runs cognitive evaluation pipeline using LCEL runnable chain
def run_cognitive_pipeline(
    cognitive_qa_json: Union[str, Dict[str, str]] = "routed_tasks.json",
) -> Dict[str, Any]:
    cognitive_agent = CognitiveAgent()

    pipeline_chain = (
        RunnableLambda(load_cognitive_input)
        | RunnableLambda(lambda qa: {
            "routed_cognitive_questions": len(qa),
            "analysis": cognitive_agent.analyze_cognitive(qa),
        })
    )

    return pipeline_chain.invoke(cognitive_qa_json)


if __name__ == "__main__":
    result = run_cognitive_pipeline()
    print(json.dumps(result, indent=2, ensure_ascii=False))
