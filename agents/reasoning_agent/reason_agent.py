import json
from typing import Any, Dict, Optional, Union
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
    qna: Union[Dict[str, Any], list] = Field(
        default_factory=dict,
        description="The reasoning question-and-answer pairs analyzed as a JSON object dictionary. Format: {\"question\": \"answer\"}.",
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
                "Analyze ONLY the following assigned reasoning question-and-answer pairs to evaluate logical reasoning quality, problem-solving approach, error patterns, and overall assessment.\n\n"
                "CRITICAL: Base your evaluation strictly on the questions and answers provided in Assigned Reasoning Tasks below. Do NOT invent, hallucinate, or assume any external questions or answers.\n\n"
                "Assigned Reasoning Tasks:\n{tasks}\n\n"
                "Return ONLY a valid JSON object without any markdown headers, conversational text, or explanations outside the JSON. Ensure 'qna' is a JSON object dictionary, not a list.\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["tasks"],
            partial_variables={"format_instructions": format_instructions},
        )
        self.chain = self.prompt | self.llm | self.parser

    # Analyzes assigned reasoning Q&A pairs and returns structured evaluation
    def analyze_reasoning(self, tasks: Dict[str, str]) -> Dict[str, Any]:
        if not tasks:
            return {
                "qna": {},
                "reasoning_result": "No reasoning-specific evaluation tasks were routed for this session.",
            }

        parsed: ReasoningAnalysisOutput = self.chain.invoke({
            "tasks": json.dumps(tasks, indent=2),
        })

        return {
            "qna": tasks,
            "reasoning_result": parsed.reasoning_result,
        }
