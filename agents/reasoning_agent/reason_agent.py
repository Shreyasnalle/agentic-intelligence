import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

try:
    from agents.questioner.models import get_hf_llm
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from agents.questioner.models import get_hf_llm


# Output schema for reasoning evaluation
class ReasoningAnalysisOutput(BaseModel):
    qna: Dict[str, Any] = Field(
        default_factory=dict,
        description="The reasoning question-and-answer pairs analyzed as a dictionary.",
    )
    reasoning_result: str = Field(
        description="A comprehensive evaluation text string covering logical reasoning quality, problem-solving approach, error patterns, and overall assessment.",
    )


# Evaluates logical reasoning and problem-solving from orchestrator-routed tasks
class ReasoningAgent:
    # Initializes reasoning agent, Pydantic output parser, prompt, and LLM chain
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm(max_new_tokens=1536)
        self.parser = PydanticOutputParser(pydantic_object=ReasoningAnalysisOutput)
        format_instructions = self.parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template=(
                "You are an expert Reasoning Analyst evaluating human cognitive and logical capabilities.\n"
                "Analyze the following assigned reasoning question-and-answer pairs to evaluate logical reasoning quality, problem-solving approach, error patterns, and overall assessment.\n\n"
                "Assigned Reasoning Tasks:\n{tasks}\n\n"
                "Return ONLY a valid JSON object without any markdown headers, conversational text, or explanations outside the JSON.\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["tasks"],
            partial_variables={"format_instructions": format_instructions},
        )
        self.chain = self.prompt | self.llm | self.parser

    # Analyzes assigned reasoning Q&A pairs and returns structured evaluation
    def analyze_reasoning(self, tasks: Dict[str, str]) -> Dict[str, Any]:
        parsed: ReasoningAnalysisOutput = self.chain.invoke({
            "tasks": json.dumps(tasks, indent=2),
        })

        return {
            "qna": parsed.qna,
            "reasoning_result": parsed.reasoning_result,
        }
