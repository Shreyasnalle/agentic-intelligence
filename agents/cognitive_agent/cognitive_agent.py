import json
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from agents.questioner.models import DEEPSEEK_REASONING_MODEL_ID, get_hf_llm
except ModuleNotFoundError:
    from questioner.models import DEEPSEEK_REASONING_MODEL_ID, get_hf_llm


class CognitiveAgent:
    """Evaluates attention, memory, and pattern-related capabilities from orchestrator-routed Q&A pairs."""

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm(repo_id=DEEPSEEK_REASONING_MODEL_ID, max_new_tokens=1536) or get_hf_llm(max_new_tokens=1536)
        self.parser = StrOutputParser()
        self.prompt = PromptTemplate(
            input_variables=["qa_json"],
            template=(
                "You are a Cognitive Analyst Agent.\n"
                "Analyze ONLY the user question-answer pairs provided below (already routed by the orchestrator).\n"
                "Your task is to evaluate the user's attention, memory, and pattern-related responses.\n\n"
                "Return a concise analysis with:\n"
                "1) Pattern recognition ability\n"
                "2) Working memory retention & tracking\n"
                "3) Attention to operational details\n"
                "4) Overall cognitive assessment\n\n"
                "Question-Answer Pairs (JSON):\n{qa_json}\n"
            ),
        )
        self.chain = None
        if self.llm is not None:
            self.chain = self.prompt | self.llm | self.parser

    def analyze_cognitive(self, qa_pairs: Dict[str, str]) -> str:
        if not qa_pairs:
            return "No routed cognitive Q&A pairs were provided for analysis."

        if self.chain is None:
            raise RuntimeError("Model or chain is not initialized.")

        return self.chain.invoke({"qa_json": json.dumps(qa_pairs, indent=2)}).strip()
