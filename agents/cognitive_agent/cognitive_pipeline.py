import json
from pathlib import Path
from typing import Dict, Any, Union

try:
    from agents.cognitive_agent.cognitive_agent import CognitiveAgent
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agents.cognitive_agent.cognitive_agent import CognitiveAgent


# Loads cognitive tasks dictionary from routed_tasks.json or input dictionary
def load_cognitive_input(source: Union[str, Dict[str, Any]] = "routed_tasks.json") -> Dict[str, str]:
    if isinstance(source, dict):
        return source.get("cognitive_agent", source)

    with open(source, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data.get("cognitive_agent", {})


# Saves the cognitive analysis results to agents_analysis.json
def save_agents_analysis(
    analysis_data: Dict[str, Any],
    output_filepath: str = "agents_analysis.json",
) -> None:
    output = {}
    path = Path(output_filepath)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as file:
                output = json.load(file)
        except Exception:
            output = {}

    output["cognitive_agent"] = analysis_data

    with open(output_filepath, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)


create_agents_analysis_json = save_agents_analysis


# Runs cognitive evaluation pipeline and saves output to agents_analysis.json
def run_cognitive_pipeline(
    source: Union[str, Dict[str, Any]] = "routed_tasks.json",
    output_filepath: str = "agents_analysis.json",
) -> Dict[str, Any]:
    cognitive_tasks = load_cognitive_input(source)
    cognitive_agent = CognitiveAgent()
    analysis = cognitive_agent.analyze_cognitive(cognitive_tasks)
    save_agents_analysis(analysis, output_filepath)
    return analysis


run = run_cognitive_pipeline


if __name__ == "__main__":
    result = run_cognitive_pipeline()
    print(json.dumps(result, indent=2, ensure_ascii=False))
