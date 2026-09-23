import json
from typing import Dict
from langchain_core.messages import HumanMessage
from agents.questioner.questioner_agent import QuestionerAgent


# Runs the multi-turn cognitive interview pipeline
def run(max_turns: int = 5, output_filepath: str = "qa_transcript.json") -> Dict[str, str]:
    questioner = QuestionerAgent(max_turns=max_turns)
    count = 0
    current_q = None
    user_input = None

    while count < max_turns:
        if count == 0:
            ai_message = questioner.generate_initial_question()
            count += 1
            current_q = ai_message["ai"]
        else:
            interaction = {"ai": current_q, "user": user_input}
            ai_message = questioner.generate_next_question(interaction)
            count += 1
            current_q = ai_message["ai"]

        print(f"AI: {current_q}")
        user_input = input("User: ").strip()

    if current_q and user_input:
        questioner.qa_history[current_q] = user_input
        questioner.memory.append(HumanMessage(content=user_input))

    final_qa = questioner.get_qa_history()

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(final_qa, f, indent=2)

    return final_qa


if __name__ == "__main__":
    run(max_turns=5)
