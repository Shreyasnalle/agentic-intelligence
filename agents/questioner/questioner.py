import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add repository root to sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.questioner.schemas import (
    QuestionOutput,
    FramedInteraction,
    QASessionResult,
)
from agents.questioner.questioner_agent import QuestionerAgent
from agents.questioner.assistant_agent import AssistantAgent


class InterviewWorkflow:
    """
    Coordinates the dialogue cycle between Questioner Agent and Assistant Agent:
    1. Questioner Agent generates a structured question.
    2. User inputs a response (interactively or simulated).
    3. Assistant Agent frames question + response into a structured JSON component.
    4. Framed interaction is passed back to Questioner Agent for subsequent turns.
    5. Outputs the final JSON dictionary mapping questions to user responses.
    """

    def __init__(
        self,
        questioner: Optional[QuestionerAgent] = None,
        assistant: Optional[AssistantAgent] = None,
        max_turns: int = 3,
    ):
        self.questioner = questioner or QuestionerAgent(max_turns=max_turns)
        self.assistant = assistant or AssistantAgent()
        self.max_turns = max_turns
        self.history: List[FramedInteraction] = []

    def run(
        self,
        interactive: bool = True,
        simulated_responses: Optional[List[str]] = None,
        output_filepath: Optional[str] = "qa_transcript.json",
    ) -> QASessionResult:
        """
        Executes the multi-agent questioner workflow.
        """
        self.history = []
        turn = 1
        sim_idx = 0

        print("=" * 60)
        print(" Starting Adaptive Cognitive Interview Session")
        print(f" Target Turns: {self.max_turns} | Open-Source HF Models (JSON Structured)")
        print("=" * 60)

        while turn <= self.max_turns:
            # 1. Questioner Agent asks next question based on history
            question_data: QuestionOutput = self.questioner.generate_question(
                turn_index=turn,
                history=self.history,
            )

            print(f"\n[Turn {turn}] Questioner Agent ({question_data.dimension_probed} | {question_data.difficulty_level}):")
            print(f"Question: {question_data.question}")

            # 2. Capture user response
            if interactive and not simulated_responses:
                user_input = input("\nYour Response: ").strip()
                if not user_input:
                    user_input = "No response provided."
            else:
                if simulated_responses and sim_idx < len(simulated_responses):
                    user_input = simulated_responses[sim_idx]
                    sim_idx += 1
                else:
                    user_input = f"Simulated logical response for turn {turn}."
                print(f"\n[Simulated Response]: {user_input}")

            # 3. Assistant Agent frames answer and question into structured JSON
            framed: FramedInteraction = self.assistant.frame_interaction(
                turn_index=turn,
                question=question_data.question,
                raw_user_response=user_input,
                dimension_probed=question_data.dimension_probed,
            )

            # 4. Store structured interaction and pass back to Questioner for next round
            self.history.append(framed)

            if question_data.is_evaluation_complete or turn >= self.max_turns:
                break

            turn += 1

        # 5. Build final output JSON: key as question, value as user response
        final_qa_pairs: Dict[str, str] = {
            item.question: item.raw_user_response for item in self.history
        }

        session_result = QASessionResult(
            total_turns=len(self.history),
            qa_pairs=final_qa_pairs,
            structured_interactions=self.history,
        )

        if output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(final_qa_pairs, f, indent=2)
            print(f"\n[✓] Final Q&A JSON successfully exported to: {output_filepath}")

        return session_result


def run_interview(
    max_turns: int = 3,
    interactive: bool = True,
    simulated_responses: Optional[List[str]] = None,
    output_filepath: Optional[str] = "qa_transcript.json",
) -> Dict[str, str]:
    """
    Convenience function to run the interview and directly return the final
    JSON dictionary where key = question and value = user's response.
    """
    workflow = InterviewWorkflow(max_turns=max_turns)
    result = workflow.run(
        interactive=interactive,
        simulated_responses=simulated_responses,
        output_filepath=output_filepath,
    )
    return result.qa_pairs


if __name__ == "__main__":
    # If --test or non-interactive flag passed, run with simulated responses
    if "--test" in sys.argv or not sys.stdin.isatty():
        sample_responses = [
            "The ball costs 5 cents. If the bat is $1.05 and the ball is $0.05, the total is $1.10 and the bat is $1.00 more.",
            "The next number is 42. The difference sequence is 4, 6, 8, 10, so adding 12 gives 42.",
            "After 4 moves from box 3, the box number must have the same parity as 3 (odd) after odd steps or same parity after even steps: 3 is odd, after 4 steps it must end on an odd box (1, 3, or 5). Since 2 is even, it is impossible to be in box 2.",
        ]
        final_json = run_interview(
            max_turns=3,
            interactive=False,
            simulated_responses=sample_responses,
        )
    else:
        final_json = run_interview(max_turns=3, interactive=True)

    print("\n" + "=" * 60)
    print("FINAL Q&A JSON (Key: Question, Value: User Response):")
    print("=" * 60)
    print(json.dumps(final_json, indent=2))
