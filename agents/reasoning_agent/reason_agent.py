from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from agents.questioner.models import get_hf_llm
except ModuleNotFoundError:
    from questioner.models import get_hf_llm


class ReasoningAgent:
    """Evaluates logical reasoning and problem-solving from orchestrator-routed Q&A pairs."""

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm()
        self.parser = StrOutputParser()
        self.prompt = PromptTemplate(
            input_variables=["qa_block"],
            template=(
                "You are a Reasoning Analyst Agent.\n"
                "Analyze ONLY the user question-answer pairs provided below (already routed by the orchestrator).\n"
                "Your task is to evaluate the user's logical reasoning and problem-solving ability.\n\n"
                "Return a concise analysis with:\n"
                "1) Logical reasoning quality\n"
                "2) Problem-solving approach\n"
                "3) Error patterns or assumptions\n"
                "4) Overall reasoning assessment\n\n"
                "Question-Answer Pairs:\n{qa_block}\n"
            ),
        )
        self.chain = None
        if self.llm is not None:
            self.chain = self.prompt | self.llm | self.parser

    def _format_qa_pairs(self, qa_pairs: Dict[str, str]) -> str:
        lines = []
        for idx, (question, answer) in enumerate(qa_pairs.items(), start=1):
            q = (question or "").strip()
            a = (answer or "").strip()
            if not q:
                continue
            if not a:
                a = "No response provided."
            lines.append(f"{idx}. Question: {q}\n   Answer: {a}")
        return "\n".join(lines)

    def analyze_reasoning(self, qa_pairs: Dict[str, str]) -> str:
        if not qa_pairs:
            return "No routed reasoning Q&A pairs were provided for analysis."

        if self.chain is None:
            raise RuntimeError("Model or chain is not initialized.")

        qa_block = self._format_qa_pairs(qa_pairs)
        if not qa_block:
            return "No valid routed reasoning Q&A pairs were provided for analysis."

        return self.chain.invoke({"qa_block": qa_block}).strip()
