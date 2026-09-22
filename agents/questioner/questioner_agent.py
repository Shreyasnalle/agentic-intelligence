import re
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser

try:
    from agents.questioner.models import get_hf_llm
except ModuleNotFoundError:
    from models import get_hf_llm

# Pydantic schemas for structured question output and session results
class QuestionMessage(BaseModel):
    ai: str = Field(description="The cognitive evaluation question asked by the counselor to the user.")

class QASessionResult(BaseModel):
    total_turns: int = Field(description="Total number of completed Q&A interactions across the interview session.")
    qa_pairs: Dict[str, str] = Field(description="Final mapping where each key is the exact evaluation question asked and value is the user's verbatim response.")

class QuestionerAgent:
    # Initializes QuestionerAgent, LLM, chains, memory, and output parser
    def __init__(self, llm: Optional[Any] = None, max_turns: int = 5):
        self.llm = llm or get_hf_llm()
        self.max_turns = max_turns
        self.current_turn = 0
        self.current_question: Optional[str] = None
        self.qa_history: Dict[str, str] = {}
        self.memory: List[BaseMessage] = []  # plain list recommended by LangChain docs
        self.parser = PydanticOutputParser(pydantic_object=QuestionMessage)
        self._build_prompts()

    # Strips think tags, code fences, and extracts first valid JSON object from LLM output
    def _clean_output(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = text.strip()
        # Extract first JSON object if model wraps output in extra text
        match = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
        if match:
            return match.group(0)
        return text

    # Builds prompt templates and compiles LCEL chains with output parser
    def _build_prompts(self):
        format_instructions = self.parser.get_format_instructions()

        self.initial_chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Counselor Agent evaluating human cognitive and reasoning capabilities.\n"
                "This is Turn 1 of {max_turns}.\n"
                "Task: Generate an engaging, calibrated opening question testing deductive logic, pattern recognition, working memory, or premise validation.\n\n"
                "{format_instructions}\n"
            ),
            (
                "human",
                "Please generate the opening cognitive evaluation question.",
            ),
        ]).partial(format_instructions=format_instructions)

        self.adaptive_chat_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert Counselor Agent conducting an adaptive cognitive interview.\n"
                "Current Turn: {turn_index} of {max_turns}.\n"
                "Review the previous questions and answers in conversation memory.\n"
                "Formulate the next evaluation question dynamically adapted to the user's prior reasoning:\n"
                "- Probe depth, test boundary conditions, or pivot to another cognitive dimension.\n"
                "- Do not repeat previous questions.\n\n"
                "{format_instructions}\n"
            ),
            MessagesPlaceholder(variable_name="history"),
            (
                "human",
                "Based on my previous answers in the dialogue history, what is your next evaluation question?",
            ),
        ]).partial(format_instructions=format_instructions)

        if self.llm is not None:
            self.initial_chain = self.initial_chat_prompt | self.llm
            self.adaptive_chain = self.adaptive_chat_prompt | self.llm
        else:
            self.initial_chain = None
            self.adaptive_chain = None

    # Generates the opening question once for Turn 1
    def generate_initial_question(self) -> Dict[str, str]:
        if self.current_turn != 0:
            raise RuntimeError("generate_initial_question can only be called once for the first question.")

        if self.initial_chain is None:
            raise RuntimeError("Model or chain is not initialized.")

        for attempt in range(3):
            raw = self.initial_chain.invoke({"max_turns": self.max_turns})
            cleaned = self._clean_output(raw.content)
            if cleaned:
                try:
                    result: QuestionMessage = self.parser.parse(cleaned)
                    question_text = result.ai.strip()
                    self.current_turn = 1
                    self.current_question = question_text
                    self.memory.append(AIMessage(content=question_text))
                    return {"ai": question_text}
                except Exception:
                    continue
        raise RuntimeError("Failed to generate initial question after 3 attempts.")

    # Generates adaptive follow-up questions for the remaining 9 turns based on user answers and memory
    def generate_next_question(
        self,
        interaction: Dict[str, str],
        memory: Optional[List[BaseMessage]] = None,
    ) -> Dict[str, str]:
        if self.current_turn < 1:
            raise RuntimeError("Initial question must be generated before generate_next_question can be called.")

        if self.current_turn >= self.max_turns:
            raise RuntimeError(f"Question limit reached: generate_next_question cannot be called more than {self.max_turns - 1} times.")

        if self.adaptive_chain is None:
            raise RuntimeError("Model or chain is not initialized.")

        active_memory = memory if memory is not None else self.memory
        ai_q = interaction.get("ai") or self.current_question
        user_a = interaction.get("user", "")

        if ai_q:
            self.qa_history[ai_q] = user_a

        has_ai_msg = any(isinstance(m, AIMessage) and m.content == ai_q for m in active_memory)
        if not has_ai_msg and ai_q:
            active_memory.append(AIMessage(content=ai_q))
        active_memory.append(HumanMessage(content=user_a))

        next_turn = self.current_turn + 1
        self.current_turn = next_turn

        for attempt in range(3):
            raw = self.adaptive_chain.invoke({
                "turn_index": next_turn,
                "max_turns": self.max_turns,
                "history": active_memory,
            })
            cleaned = self._clean_output(raw.content)
            if cleaned:
                try:
                    result: QuestionMessage = self.parser.parse(cleaned)
                    next_question_text = result.ai.strip()
                    self.current_question = next_question_text
                    active_memory.append(AIMessage(content=next_question_text))
                    return {"ai": next_question_text}
                except Exception:
                    continue
        raise RuntimeError("Failed to generate next question after 3 attempts.")

    # Returns the accumulated dictionary of questions mapped to user responses
    def get_qa_history(self) -> Dict[str, str]:
        return self.qa_history
