import json
from typing import Dict, List, Optional

try:
    from agents.reasoning_agent.reason_agent import ReasoningAgent
except ModuleNotFoundError:
    from reason_agent import ReasoningAgent


class ReasoningPipeline:
    """Loads transcript, selects routed questions, and runs reasoning analysis."""

    def __init__(self, reasoning_agent: Optional[ReasoningAgent] = None):
        self.reasoning_agent = reasoning_agent or ReasoningAgent()

    def load_qa_transcript(self, filepath: str = "qa_transcript.json") -> Dict[str, str]:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError("Q&A transcript must be a JSON object mapping questions to answers.")
        return {str(k): str(v) for k, v in data.items()}

    def select_reasoning_questions(
        self,
        qa_pairs: Dict[str, str],
        selected_questions: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        if selected_questions:
            selected_set = {q.strip() for q in selected_questions if q and q.strip()}
            return {q: a for q, a in qa_pairs.items() if q in selected_set}

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

    def run(
        self,
        transcript_path: str = "qa_transcript.json",
        selected_questions: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        qa_pairs = self.load_qa_transcript(transcript_path)
        routed_pairs = self.select_reasoning_questions(qa_pairs, selected_questions)
        analysis = self.reasoning_agent.analyze_reasoning(routed_pairs)

        return {
            "total_questions": len(qa_pairs),
            "routed_reasoning_questions": len(routed_pairs),
            "analysis": analysis,
        }


def run_reasoning_pipeline(
    transcript_path: str = "qa_transcript.json",
    selected_questions: Optional[List[str]] = None,
) -> Dict[str, object]:
    pipeline = ReasoningPipeline()
    return pipeline.run(transcript_path=transcript_path, selected_questions=selected_questions)


if __name__ == "__main__":
    result = run_reasoning_pipeline()
    print(json.dumps(result, indent=2, ensure_ascii=False))
