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


# Output schema for cognitive evaluation
class CognitiveAnalysisOutput(BaseModel):
    qna: Dict[str, Any] = Field(
        default_factory=dict,
        description="The cognitive question-and-answer pairs analyzed as a JSON object dictionary. Format: {\"question\": \"answer\"}.",
    )
    cognitive_result: str = Field(
        description="A comprehensive evaluation text string covering pattern recognition ability, working memory retention and tracking, attention to detail, and overall cognitive assessment.",
    )


# Evaluates attention, memory, and pattern-related capabilities from orchestrator-routed tasks
class CognitiveAgent:
    # Initializes cognitive agent, Pydantic output parser, prompt, and LLM chain
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm(max_new_tokens=1536)
        self.parser = PydanticOutputParser(pydantic_object=CognitiveAnalysisOutput)
        format_instructions = self.parser.get_format_instructions()
        self.prompt = PromptTemplate(
            template=(
                "You are an expert Cognitive Analyst evaluating human cognitive capabilities.\n"
                "Analyze the following assigned cognitive question-and-answer pairs to evaluate pattern recognition ability, working memory retention and tracking, attention to detail, and overall cognitive assessment.\n\n"
                "Assigned Cognitive Tasks:\n{tasks}\n\n"
                "Return ONLY a valid JSON object without any markdown headers, conversational text, or explanations outside the JSON. Ensure 'qna' is a JSON object dictionary, not a list.\n\n"
                "{format_instructions}\n"
            ),
            input_variables=["tasks"],
            partial_variables={"format_instructions": format_instructions},
        )
        self.chain = self.prompt | self.llm | self.parser

    # Analyzes assigned cognitive Q&A pairs and returns structured evaluation
    def analyze_cognitive(self, tasks: Dict[str, str]) -> Dict[str, Any]:
        parsed: CognitiveAnalysisOutput = self.chain.invoke({
            "tasks": json.dumps(tasks, indent=2),
        })

        return {
            "qna": parsed.qna,
            "cognitive_result": parsed.cognitive_result,
        }
