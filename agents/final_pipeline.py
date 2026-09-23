import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnableSequence,
)

try:
    from agents.questioner.questioner_pipeline import run as run_questioner
    from agents.agent_orchestrator.orchestrator import OrchestratorAgent
    from agents.reasoning_agent.reason_agent import ReasoningAgent
    from agents.cognitive_agent.cognitive_agent import CognitiveAgent
    from agents.final_report.report_agent import ReportAgent
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from agents.questioner.questioner_pipeline import run as run_questioner
    from agents.agent_orchestrator.orchestrator import OrchestratorAgent
    from agents.reasoning_agent.reason_agent import ReasoningAgent
    from agents.cognitive_agent.cognitive_agent import CognitiveAgent
    from agents.final_report.report_agent import ReportAgent


# Step 1: Loads existing Q&A transcript or runs interactive Questioner Agent
def step_interview_or_load(source: Optional[Union[str, Dict[str, str]]] = None) -> Dict[str, str]:
    if isinstance(source, dict):
        return source

    if isinstance(source, str) and Path(source).exists():
        with open(source, "r", encoding="utf-8") as file:
            return json.load(file)

    return run_questioner(max_turns=5, output_filepath="qa_transcript.json")


# Step 2: Partitions Q&A transcript between specialist agents via Agent Orchestrator
def step_orchestrate(qa_data: Dict[str, str]) -> Dict[str, Any]:
    orchestrator = OrchestratorAgent()
    result = orchestrator.orchestrate(qa_data)

    routed_output = {
        "reasoning_agent": result.get("reasoning_agent", {}),
        "cognitive_agent": result.get("cognitive_agent", {}),
    }

    with open("routed_tasks.json", "w", encoding="utf-8") as file:
        json.dump(routed_output, file, indent=2, ensure_ascii=False)

    return routed_output


# Step 3a: Executes Reasoning Agent evaluation on allocated tasks
def step_reasoning(routed_tasks: Dict[str, Any]) -> Dict[str, Any]:
    tasks = routed_tasks.get("reasoning_agent", {})
    agent = ReasoningAgent()
    return agent.analyze_reasoning(tasks)


# Step 3b: Executes Cognitive Agent evaluation on allocated tasks
def step_cognitive(routed_tasks: Dict[str, Any]) -> Dict[str, Any]:
    tasks = routed_tasks.get("cognitive_agent", {})
    agent = CognitiveAgent()
    return agent.analyze_cognitive(tasks)


# Step 4: Persists combined specialist evaluations to agents_analysis.json
def step_save_analysis(specialist_analyses: Dict[str, Any]) -> Dict[str, Any]:
    with open("agents_analysis.json", "w", encoding="utf-8") as file:
        json.dump(specialist_analyses, file, indent=2, ensure_ascii=False)

    return specialist_analyses


# Step 5: Generates final cognitive intelligence profile report via Report Agent
def step_report(analysis_data: Dict[str, Any]) -> str:
    agent = ReportAgent()
    report = agent.generate_report(analysis_data)

    with open("final_report.md", "w", encoding="utf-8") as file:
        file.write(report)

    return report


# Builds the end-to-end multi-agent workflow runnable chain
def build_final_pipeline() -> RunnableSequence:
    return (
        RunnableLambda(step_interview_or_load)
        | RunnableLambda(step_orchestrate)
        | RunnableParallel({
            "reasoning_agent": RunnableLambda(step_reasoning),
            "cognitive_agent": RunnableLambda(step_cognitive),
        })
        | RunnableLambda(step_save_analysis)
        | RunnableLambda(step_report)
    )


# Executes the complete end-to-end multi-agent evaluation pipeline
def run(source: Optional[Union[str, Dict[str, str]]] = "qa_transcript.json") -> str:
    pipeline = build_final_pipeline()
    return pipeline.invoke(source)


if __name__ == "__main__":
    final_report = run()
    print(final_report)
