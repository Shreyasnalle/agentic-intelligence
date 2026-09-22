import sys
import json
from typing import Dict, List, Optional

try:
    from agents.questioner.questioner_agent import QuestionerAgent
except ModuleNotFoundError:
    from questioner_agent import QuestionerAgent


# Manages the multi-turn interview workflow between counselor agent and user
class InterviewWorkflow:
    # Initializes the interview workflow with QuestionerAgent and turn limit
    def __init__(
        self,
        questioner: Optional[QuestionerAgent] = None,
        max_turns: int = 5,
    ):
        self.questioner = questioner or QuestionerAgent(max_turns=max_turns)
        self.max_turns = max_turns

    # Executes the interview workflow calling questioner methods in sequential order
    def run(
        self,
        interactive: bool = True,
        simulated_responses: Optional[List[str]] = None,
        output_filepath: Optional[str] = "qa_transcript.json",
    ) -> Dict[str, str]:
        sim_idx = 0
        ai_message = self.questioner.generate_initial_question()
        current_q = ai_message["ai"]

        for turn in range(1, self.max_turns + 1):
            if interactive and not simulated_responses:
                print(f"AI: {current_q}")
                try:
                    user_input = input("User: ").strip()
                except (EOFError, KeyboardInterrupt):
                    user_input = "Session ended by user."
                if not user_input:
                    user_input = "No response provided."
            else:
                if simulated_responses and sim_idx < len(simulated_responses):
                    user_input = simulated_responses[sim_idx]
                    sim_idx += 1
                else:
                    user_input = f"Simulated response for question {turn}."

                interaction = {"ai": current_q, "user": user_input}

                if turn < self.max_turns:
                    ai_message = self.questioner.generate_next_question(
                        interaction=interaction,
                        memory=self.questioner.memory,
                    )
                    current_q = ai_message["ai"]
                else:
                    self.questioner.qa_history[current_q] = user_input
                    if hasattr(self.questioner.memory, "append"):
                        from langchain_core.messages import HumanMessage
                        self.questioner.memory.append(HumanMessage(content=user_input))
                    elif hasattr(self.questioner.memory, "add_user_message"):
                        self.questioner.memory.add_user_message(user_input)

        final_qa = self.questioner.get_qa_history()

        if output_filepath:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(final_qa, f, indent=2)

        return final_qa


# Convenience function to initialize and run the interview workflow
def run_interview(
    max_turns: int = 5,
    interactive: bool = True,
    simulated_responses: Optional[List[str]] = None,
    output_filepath: Optional[str] = "qa_transcript.json",
) -> Dict[str, str]:
    workflow = InterviewWorkflow(max_turns=max_turns)
    return workflow.run(
        interactive=interactive,
        simulated_responses=simulated_responses,
        output_filepath=output_filepath,
    )


# Runs the interview when invoked directly
if __name__ == "__main__":
    if "--test" in sys.argv or not sys.stdin.isatty():
        sample_responses = [
            "The ball costs 5 cents.",
            "The next number is 42.",
        ]
        final_json = run_interview(
            max_turns=2,
            interactive=False,
            simulated_responses=sample_responses,
        )
    else:
        final_json = run_interview(max_turns=5, interactive=True)

    print(json.dumps(final_json, indent=2))

