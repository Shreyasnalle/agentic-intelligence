import json
from typing import Any, Dict, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from agents.questioner.models import get_hf_llm
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agents.questioner.models import get_hf_llm


# Generates comprehensive cognitive intelligence report from specialist agent evaluations
class ReportAgent:
    # Initializes report agent, prompt template, StrOutputParser, and LLM chain
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm(max_new_tokens=2048)
        self.parser = StrOutputParser()
        self.prompt = PromptTemplate(
            template=(
                "You are an expert Profile Analyst and Cognitive Assessment Report Agent.\n"
                "Synthesize the following specialist agent analyses (Reasoning Analyst and Cognitive Analyst) into a comprehensive, professional intelligence profile of the user.\n\n"
                "Specialist Analyses:\n"
                "{analysis_json}\n\n"
                "Generate a detailed final cognitive assessment report structured with:\n"
                "1. Executive Summary & Profile Persona\n"
                "2. Reasoning & Logical Capabilities (Strengths, Deductive/Inductive logic, Error Patterns)\n"
                "3. Cognitive & Attentional Capabilities (Working memory, Pattern recognition, Attention to detail)\n"
                "4. Synthesized Strengths & Blind Spots\n"
                "5. Overall Assessment & Development Recommendations\n"
            ),
            input_variables=["analysis_json"],
        )
        self.chain = self.prompt | self.llm | self.parser

    # Generates final comprehensive report from specialist evaluations dictionary
    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        return self.chain.invoke({
            "analysis_json": json.dumps(analysis_data, indent=2),
        }).strip()
