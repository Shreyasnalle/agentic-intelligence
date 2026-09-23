import json
from typing import Dict, Optional, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END

from agents.questioner.models import get_hf_llm


# Output schema mapping questions to reasoning and cognitive specialist agents
class OrchestratorOutput(BaseModel):
    reasoning_agent: Dict[str, str] = Field(
        default_factory=dict,
        description="Q&A pairs allocated to the Reasoning Analyst (for deductive/inductive logic, critical thinking, syllogisms, or premise validation). Format: {question: answer}.",
    )
    cognitive_agent: Dict[str, str] = Field(
        default_factory=dict,
        description="Q&A pairs allocated to the Cognitive Analyst (for pattern recognition, sequence reasoning, working memory, attention to detail). Format: {question: answer}.",
    )


# State schema tracking raw Q&A and routed agent tasks in LangGraph
class OrchestratorState(TypedDict, total=False):
    raw_qa: Dict[str, str]
    reasoning_agent: Dict[str, str]
    cognitive_agent: Dict[str, str]
    summary: Dict[str, Any]


# Agent orchestrator coordinating task classification and routing
class OrchestratorAgent:
    # Initializes orchestrator, LLM chain, parser, and compiled LangGraph workflow
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm or get_hf_llm(max_new_tokens=1024)
        self.parser = PydanticOutputParser(pydantic_object=OrchestratorOutput)
        self._build_chain()
        self._build_graph()

    # Builds LCEL prompt and classification chain with Pydantic parser
    def _build_chain(self):
        format_instructions = self.parser.get_format_instructions()

        template = (
            "You are an AI Agent Orchestrator in a multi-agent cognitive assessment system.\n"
            "Directly partition the following Q&A transcript between two specialist agents:\n\n"
            "- reasoning_agent: questions evaluating deductive logic, inductive reasoning, syllogisms, premise validation, critical thinking.\n"
            "- cognitive_agent: questions evaluating pattern recognition, numerical/spatial sequences, working memory, attention to detail.\n\n"
            "Input Q&A JSON:\n"
            "{qa_json}\n\n"
            "{format_instructions}\n"
        )

        prompt = PromptTemplate.from_template(template).partial(
            format_instructions=format_instructions
        )

        if self.llm is not None:
            self.chain = prompt | self.llm | self.parser
        else:
            self.chain = None

    # Classifies raw Q&A JSON into reasoning and cognitive agent task mappings
    def _classify_node(self, state: OrchestratorState) -> Dict[str, Any]:
        raw_qa = state.get("raw_qa", {})
        if not raw_qa:
            return {"reasoning_agent": {}, "cognitive_agent": {}}

        parsed: OrchestratorOutput = self.chain.invoke({
            "qa_json": json.dumps(raw_qa, indent=2),
        })

        return {
            "reasoning_agent": parsed.reasoning_agent,
            "cognitive_agent": parsed.cognitive_agent,
        }

    # Tallies task distribution counts for the orchestration summary
    def _route_node(self, state: OrchestratorState) -> Dict[str, Any]:
        reasoning = state.get("reasoning_agent", {})
        cognitive = state.get("cognitive_agent", {})

        summary = {
            "total_tasks": len(reasoning) + len(cognitive),
            "reasoning_tasks_count": len(reasoning),
            "cognitive_tasks_count": len(cognitive),
        }

        return {"summary": summary}

    # Constructs and compiles the LangGraph state machine
    def _build_graph(self):
        workflow = StateGraph(OrchestratorState)
        workflow.add_node("classify", self._classify_node)
        workflow.add_node("route", self._route_node)

        workflow.add_edge(START, "classify")
        workflow.add_edge("classify", "route")
        workflow.add_edge("route", END)

        self.graph = workflow.compile()

    # Executes the LangGraph workflow with input Q&A transcript
    def orchestrate(self, raw_qa: Dict[str, str]) -> Dict[str, Any]:
        return self.graph.invoke({"raw_qa": raw_qa})
