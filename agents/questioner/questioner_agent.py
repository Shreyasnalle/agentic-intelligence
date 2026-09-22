import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.questioner.schemas import QuestionOutput, FramedInteraction
from agents.models import get_hf_llm


class QuestionerAgent:
    """
    Counselor Agent that dynamically crafts evaluation questions to assess
    a user's cognitive, analytical, and reasoning capabilities using Hugging Face models.
    Outputs strictly structured QuestionOutput JSON components.
    """

    def __init__(self, llm: Optional[Any] = None, max_turns: int = 3):
        self.llm = llm or get_hf_llm()
        self.max_turns = max_turns
        if self.llm is not None:
            try:
                self.structured_llm = self.llm.with_structured_output(QuestionOutput)
            except Exception:
                self.structured_llm = None
        else:
            self.structured_llm = None

    def generate_question(
        self,
        turn_index: int,
        history: List[FramedInteraction],
    ) -> QuestionOutput:
        """
        Generates the next adaptive evaluation question based on prior framed interactions.
        """
        is_final = turn_index >= self.max_turns

        if self.structured_llm is not None:
            history_json = [item.model_dump() for item in history]
            prompt = (
                "You are an expert Counselor Agent evaluating cognitive and analytical abilities.\n"
                f"Current Turn: {turn_index} of {self.max_turns}.\n"
                f"Previous Interaction History (JSON format):\n{json.dumps(history_json, indent=2)}\n\n"
                "Instructions:\n"
                "1. If this is Turn 1, start with an engaging cognitive or reasoning challenge (e.g., deductive puzzle, pattern sequence, or premise analysis).\n"
                "2. If prior history exists, dynamically adapt your question based on how the user reasoned, responded, or solved previous questions. Probe their depth, test boundary conditions, or pivot to another dimension (e.g., working memory, deductive logic, sequential reasoning).\n"
                f"3. Mark 'is_evaluation_complete' as True if turn_index >= {self.max_turns} or if sufficient cognitive evaluation data has been gathered.\n"
                "4. Return the result strictly using the QuestionOutput structure."
            )
            try:
                result = self.structured_llm.invoke(prompt)
                if isinstance(result, QuestionOutput):
                    result.turn_index = turn_index
                    if is_final:
                        result.is_evaluation_complete = True
                    return result
            except Exception as e:
                print(f"[QuestionerAgent] Warning: LLM invocation failed ({e}), using calibrated fallback question.")

        return self._get_fallback_question(turn_index, history, is_final)

    def _get_fallback_question(
        self,
        turn_index: int,
        history: List[FramedInteraction],
        is_final: bool,
    ) -> QuestionOutput:
        fallback_questions = [
            (
                "A bat and a baseball ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost, and what is your step-by-step reasoning?",
                "deductive_logic",
                "easy",
            ),
            (
                "Consider the sequence: 2, 6, 12, 20, 30, ... What is the next number in the sequence, and what mathematical pattern or formula governs it?",
                "pattern_recognition",
                "medium",
            ),
            (
                "Imagine five boxes in a row labeled 1 to 5. A coin starts in box 3. Every minute, the coin moves to an adjacent box (either left or right). Can the coin be in box 2 after exactly 4 minutes? Explain your reasoning.",
                "working_memory_and_sequential_reasoning",
                "hard",
            ),
            (
                "If all quips are quirks, and some quirks are quacks, does it logically follow that some quips are quacks? Validate or refute the conclusion with formal logic.",
                "premise_validation",
                "hard",
            ),
        ]
        idx = min(turn_index - 1, len(fallback_questions) - 1)
        q_text, dim, diff = fallback_questions[idx]

        return QuestionOutput(
            question=q_text,
            dimension_probed=dim,
            difficulty_level=diff,
            turn_index=turn_index,
            is_evaluation_complete=is_final,
            rationale="Baseline cognitive probe.",
        )


if __name__ == "__main__":
    agent = QuestionerAgent()
    q = agent.generate_question(1, [])
    print(f"Generated Question ({q.dimension_probed}): {q.question}")
