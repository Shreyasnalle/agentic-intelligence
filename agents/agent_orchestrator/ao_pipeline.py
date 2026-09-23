import json
from typing import Dict, Any, Union

try:
    from agents.agent_orchestrator.orchestrator import OrchestratorAgent
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agents.agent_orchestrator.orchestrator import OrchestratorAgent


# Runs the agent orchestrator pipeline to partition Q&A and export routed tasks
def run(
    transcript_path: Union[str, Dict[str, str]] = "qa_transcript.json",
    output_filepath: str = "routed_tasks.json",
) -> Dict[str, Any]:
    if isinstance(transcript_path, dict):
        qa_data = transcript_path
    else:
        with open(transcript_path, "r", encoding="utf-8") as file:
            qa_data = json.load(file)

    orchestrator = OrchestratorAgent()
    result = orchestrator.orchestrate(qa_data)

    routed_output = {
        "reasoning_agent": result.get("reasoning_agent", {}),
        "cognitive_agent": result.get("cognitive_agent", {}),
    }

    with open(output_filepath, "w", encoding="utf-8") as file:
        json.dump(routed_output, file, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2, ensure_ascii=False))
