import json
from typing import Dict, Optional, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage

from agents.models import get_hf_llm
from agents.questioner.schemas import QuestionOutput


class QuestionerAgent:
    """
    Questioner (Counselor) Agent built strictly with standard LangChain components:
    - Model: Loaded from agents.models (get_hf_llm)
    - Structured Output: PydanticOutputParser(pydantic_object=QuestionOutput)
    - Chain: prompt | llm | parser
    - Memory: InMemoryChatMessageHistory
    """

    def __init__(self, llm: Optional[Any] = None, max_turns: int = 10):
        self.llm = llm or get_hf_llm()
        self.max_turns = max_turns
        self.current_turn = 1
        self.current_question: Optional[str] = None

        # Key-value history mapping: { question: user_response }
        self.qa_history: Dict[str, str] = {}

        # LangChain Memory Component
        self.memory = InMemoryChatMessageHistory()

        # LangChain Pydantic Output Parser
        self.parser = PydanticOutputParser(pydantic_object=QuestionOutput)

        # Build chains: prompt | llm | parser
        self._build_chains()

    def _build_chains(self):
        """
        Builds standard LangChain pipelines: prompt | llm | parser
        """
        format_instructions = self.parser.get_format_instructions()

        # 1. Initial Question Prompt & Chain (Turn 1)
        self.initial_prompt = PromptTemplate(
            template=(
                "You are an expert Counselor Agent evaluating human cognitive, analytical, and problem-solving abilities.\n"
                "This is Turn 1 of {max_turns}.\n\n"
                "Task:\n"
                "Generate an engaging, calibrated opening question to evaluate one of the key cognitive dimensions:\n"
                "- Deductive reasoning\n"
                "- Pattern recognition\n"
                "- Working memory and sequential reasoning\n"
                "- Critical premise validation\n\n"
                "Ensure the question is thought-provoking, specific, and clear.\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["max_turns"],
            partial_variables={"format_instructions": format_instructions},
        )

        # 2. Credibility Check & Reframing Prompt & Chain (Turns 2 to 10)
        self.reframe_prompt = PromptTemplate(
            template=(
                "You are the senior Counselor Agent in a cognitive evaluation interview.\n"
                "Current Turn: {turn_index} of {max_turns}.\n\n"
                "Conversation Memory Transcript:\n"
                "{memory_transcript}\n\n"
                "Accumulated Q&A Key-Value History:\n"
                "{qa_history_json}\n\n"
                "Assistant Agent's Proposed Next Question:\n"
                "\"{proposed_question}\" (Proposed Dimension: {proposed_dimension})\n\n"
                "Your Tasks:\n"
                "1. Credibility Check: Evaluate the assistant's proposed question against the conversation memory. Ensure it is logically sound, non-repetitive, and relevant based on prior answers.\n"
                "2. Reframe & Sharpen: Reframe and polish the question into a high-standard evaluation challenge.\n"
                "3. Set 'is_evaluation_complete' to True if turn_index >= {max_turns}, else False.\n\n"
                "{format_instructions}\n"
            ),
            input_variables=[
                "turn_index",
                "max_turns",
                "memory_transcript",
                "qa_history_json",
                "proposed_question",
                "proposed_dimension",
            ],
            partial_variables={"format_instructions": format_instructions},
        )

        if self.llm is not None:
            # Standard LangChain LCEL pipeline: prompt | llm | parser
            self.initial_chain = self.initial_prompt | self.llm | self.parser
            self.reframe_chain = self.reframe_prompt | self.llm | self.parser
        else:
            self.initial_chain = None
            self.reframe_chain = None

    def get_memory_transcript(self) -> str:
        """
        Extracts dialogue transcript from the LangChain Memory component.
        """
        lines = []
        for msg in self.memory.messages:
            prefix = "Counselor (AI):" if isinstance(msg, AIMessage) else "User:"
            lines.append(f"{prefix} {msg.content}")
        return "\n".join(lines) if lines else "No prior conversation in memory."

    def generate_initial_question(self) -> QuestionOutput:
        """
        Generates opening Question 1 using: prompt | llm | parser
        """
        if self.initial_chain is None:
            raise RuntimeError(
                "[QuestionerAgent] Model/Chain is not initialized. Please ensure HUGGINGFACEHUB_API_TOKEN is set."
            )

        output: QuestionOutput = self.initial_chain.invoke({"max_turns": self.max_turns})
        output.turn_index = 1
        output.is_evaluation_complete = (1 >= self.max_turns)

        self.current_turn = 1
        self.current_question = output.question
        self.memory.add_ai_message(output.question)

        return output

    def process_user_response(
        self,
        user_answer: str,
        question: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Takes user's answer, pairs it with the question into { question: user_answer },
        records it in memory, and returns the key-value dictionary to pass to assistant_agent.
        """
        active_question = question or self.current_question
        if not active_question:
            raise ValueError("[QuestionerAgent] No active question found to pair with user response.")

        self.memory.add_user_message(user_answer)
        self.qa_history[active_question] = user_answer

        return {active_question: user_answer}

    def recheck_and_reframe_question(
        self,
        assistant_question: str,
        assistant_dimension: Optional[str] = "general",
        turn_index: Optional[int] = None,
    ) -> QuestionOutput:
        """
        Receives candidate question from assistant_agent, checks credibility against
        memory, reframes it, and outputs the next question using: prompt | llm | parser
        """
        if self.reframe_chain is None:
            raise RuntimeError(
                "[QuestionerAgent] Model/Chain is not initialized. Please ensure HUGGINGFACEHUB_API_TOKEN is set."
            )

        next_turn = turn_index if turn_index is not None else (self.current_turn + 1)
        is_final = next_turn >= self.max_turns

        payload = {
            "turn_index": next_turn,
            "max_turns": self.max_turns,
            "memory_transcript": self.get_memory_transcript(),
            "qa_history_json": json.dumps(self.qa_history, indent=2),
            "proposed_question": assistant_question,
            "proposed_dimension": assistant_dimension or "general",
        }

        output: QuestionOutput = self.reframe_chain.invoke(payload)
        output.turn_index = next_turn
        output.is_evaluation_complete = is_final

        self.current_turn = next_turn
        self.current_question = output.question
        self.memory.add_ai_message(output.question)

        return output

    def generate_question(
        self,
        turn_index: int,
        qa_history: Optional[Dict[str, str]] = None,
        proposed_question: Optional[str] = None,
        proposed_dimension: Optional[str] = None,
    ) -> QuestionOutput:
        """
        Unified dispatch method across all 10 turns.
        """
        if qa_history:
            self.qa_history.update(qa_history)

        if turn_index == 1 or not proposed_question:
            return self.generate_initial_question()
        else:
            return self.recheck_and_reframe_question(
                assistant_question=proposed_question,
                assistant_dimension=proposed_dimension,
                turn_index=turn_index,
            )
