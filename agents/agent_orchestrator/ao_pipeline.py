import json
from typing import Dict, TypedDict

from langgraph.graph import END, START, StateGraph

try:
    from agents.reasoning_agent.reason_agent import ReasoningAgent
except ModuleNotFoundError:
    from reasoning_agent.reason_agent import ReasoningAgent


class OrchestratorState(TypedDict, total=False):
    qa_pairs: Dict[str, str]
    reasoning_questions: Dict[str, str]
    reasoning_analysis: str


def _route_reasoning_questions(qa_pairs: Dict[str, str]) -> Dict[str, str]:
    keywords = [
        "why",
        "how",
        "reason",
        "logic",
        "deduct",
        "induct",
        "pattern",
        "sequence",
        "explain",
        "solve",
        "proof",
    ]
    routed = {
        q: a
        for q, a in qa_pairs.items()
        if any(keyword in q.lower() for keyword in keywords)
    }
    return routed if routed else qa_pairs


def route_questions_node(state: OrchestratorState) -> OrchestratorState:
    qa_pairs = state.get("qa_pairs", {})
    return {"reasoning_questions": _route_reasoning_questions(qa_pairs)}


def reasoning_agent_node(state: OrchestratorState) -> OrchestratorState:
    agent = ReasoningAgent()
    reasoning_questions = state.get("reasoning_questions", {})
    analysis = agent.analyze_reasoning(reasoning_questions)
    return {"reasoning_analysis": analysis}


def build_orchestrator_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("route_questions", route_questions_node)
    graph.add_node("reasoning_analysis", reasoning_agent_node)

    graph.add_edge(START, "route_questions")
    graph.add_edge("route_questions", "reasoning_analysis")
    graph.add_edge("reasoning_analysis", END)

    return graph.compile()


def run_orchestrator(qa_pairs: Dict[str, str]) -> OrchestratorState:
    app = build_orchestrator_graph()
    initial_state: OrchestratorState = {"qa_pairs": qa_pairs}
    return app.invoke(initial_state)


if __name__ == "__main__":
    with open("qa_transcript.json", "r", encoding="utf-8") as file:
        qa_data = json.load(file)

    output = run_orchestrator(qa_data)
    print(json.dumps(output, indent=2, ensure_ascii=False))
