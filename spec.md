> **Important Note:** Don't write unwanted block of code, fallback functions, possibilites run and unnnessary print statements, unless and until specified in the prompt

# AGENTS FOLDER

## QUESTIONER AGENT

### 1. Execution Workflow
The Questioner Agent conducts an orderly, multi-turn cognitive interview between the Counselor AI and the User across 5 total turns:

1. **Initialization (`run` in `questioner_pipeline.py`)**:
   - `QuestionerAgent(max_turns=5)` is initialized with turn counter `count = 0`, an empty message memory list `self.memory = []`, and an empty Q&A history dictionary `self.qa_history = {}`.

2. **Turn 1 — Initial Question Generation**:
   - Condition: `count == 0`.
   - `questioner.generate_initial_question()` is called **exactly once**.
   - Invokes the `initial_chain` (`PromptTemplate | LLM | PydanticOutputParser`).
   - The opening question is saved to `self.memory` as an `AIMessage`.
   - `count` is incremented to `1`.
   - The question is displayed via `print(f"AI: {current_q}")`.
   - User response is collected via `user_input = input("User: ")`.

3. **Turns 2 to 5 — Adaptive Follow-Up Questions**:
   - Condition: `count > 0` and `count < max_turns`.
   - `questioner.generate_next_question(interaction)` is called **exactly 4 times**.
   - Each call:
     - Stores the previous Q&A pair in `self.qa_history`.
     - Appends the user's response as a `HumanMessage` to `self.memory`.
     - Increments `count` by 1.
     - Invokes `adaptive_chain` (`ChatPromptTemplate with MessagesPlaceholder | LLM | PydanticOutputParser`), passing the accumulated `self.memory` as conversation context.
     - Appends the newly generated follow-up question as an `AIMessage` to `self.memory`.
     - Displays `AI: {current_q}` and collects `user_input = input("User: ")`.

4. **Finalization & Handoff**:
   - The loop terminates once `count == max_turns` (5 questions generated).
   - The 5th user answer is appended to `self.qa_history` and `self.memory`.
   - `self.get_qa_history()` returns the complete `{ question: answer }` mapping.
   - The pipeline writes the dictionary to `qa_transcript.json` as the input bridge for downstream agents.

```
[run() entry] ──► count = 0
       │
       ▼
┌────────────────────────────────────────────────────────────────────────┐
│ while count < max_turns (5):                                           │
│                                                                        │
│   if count == 0:                                                       │
│       generate_initial_question() [Turn 1: PromptTemplate | LLM]       │
│       count = 1                                                        │
│                                                                        │
│   else:                                                                │
│       generate_next_question(interaction) [Turns 2-5: Memory + LLM]   │
│       count += 1                                                       │
│                                                                        │
│   Display: AI: {current_q}                                             │
│   Collect: User: {user_input}                                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (After Turn 5 response)
                                    ▼
       Store 5th response into qa_history & memory
                                    │
                                    ▼
       Export qa_history to qa_transcript.json
```

---

### 2. File & Component Breakdown (with LangChain Components)

#### A. `models.py`
**Purpose**: Centralized factory initializing LLM connections.

- **Functions**:
  - `get_hf_llm(repo_id, api_token, temperature, max_new_tokens)`: Instantiates the chat model.
- **LangChain Components Used**:
  - **`HuggingFaceEndpoint`** (`langchain_huggingface`):
    - *Where used*: Inside `get_hf_llm` to establish a direct connection to Hugging Face's Serverless Inference API.
    - *Why used*: Provides an interface to run open-weight models (e.g., `meta-llama/Llama-3.1-8B-Instruct`) without local GPU requirements.
  - **`ChatHuggingFace`** (`langchain_huggingface`):
    - *Where used*: Wraps the raw `HuggingFaceEndpoint` instance.
    - *Why used*: Adapts completion endpoints into a LangChain Chat Model that accepts structured message objects (`AIMessage`, `HumanMessage`) and chat prompts.

---

#### B. `questioner_agent.py`
**Purpose**: Houses the `QuestionerAgent` class responsible for state tracking, prompt compilation, and question generation.

- **Classes & Schemas**:
  - `QuestionMessage(BaseModel)`:
    - *Purpose*: Defines output schema (`ai: str = Field(...)`).
    - *LangChain Component*: **`PydanticOutputParser`**
      - *Where used*: `self.parser = PydanticOutputParser(pydantic_object=QuestionMessage)`, bound into both LCEL chains.
      - *Why used*: Guarantees that the LLM returns valid structured JSON matching `{"ai": "<question>"}`, preventing parsing failures or unstructured model chatter.

- **Methods**:
  - `__init__(llm, max_turns=5)`:
    - *Purpose*: Initializes turn count, memory list, history mapping, output parser, and calls `_build_prompts()`.
    - *Where called*: By `run()` in `questioner_pipeline.py`.
    - *LangChain Component*: **`BaseMessage`**
      - *Where used*: `self.memory: List[BaseMessage] = []`.
      - *Why used*: Recommended LangChain standard for storing conversation history as a typed, ordered list of messages.

  - `_build_prompts()`:
    - *Purpose*: Constructs the prompt templates and compiles the LCEL runnable chains.
    - *Where called*: Automatically inside `__init__`.
    - *LangChain Components Used*:
      - **`PromptTemplate`** (`langchain_core.prompts`):
        - *Where used*: `self.initial_chat_prompt` for Turn 1.
        - *Why used*: Turn 1 has no previous conversation history. A direct `PromptTemplate` is cleaner and eliminates unnecessary chat role overhead.
      - **`ChatPromptTemplate`** (`langchain_core.prompts`):
        - *Where used*: `self.adaptive_chat_prompt` for Turns 2–5.
        - *Why used*: Needed for multi-turn conversations to separate the static counselor persona from dynamic dialogue history.
      - **`MessagesPlaceholder`** (`langchain_core.prompts`):
        - *Where used*: Inside `self.adaptive_chat_prompt` with `variable_name="history"`.
        - *Why used*: Dynamically injects the full conversation memory (`self.memory`) into the prompt before the final human prompt, allowing the LLM to inspect all prior answers.
      - **LCEL Pipe Operator (`|`)**:
        - *Where used*:
          - `self.initial_chain = self.initial_chat_prompt | self.llm | self.parser`
          - `self.adaptive_chain = self.adaptive_chat_prompt | self.llm | self.parser`
        - *Why used*: Connects prompt, model, and parser into a single executable pipeline where `chain.invoke(...)` parses the model output directly into a validated `QuestionMessage` in one step.

  - `generate_initial_question() -> Dict[str, str]`:
    - *Purpose*: Generates the opening calibrated cognitive question.
    - *Where called*: In `questioner_pipeline.py` when `count == 0` (Turn 1 only).
    - *LangChain Component*: **`AIMessage`**
      - *Where used*: `self.memory.append(AIMessage(content=question_text))`.
      - *Why used*: Records the first AI question into the conversation memory.

  - `generate_next_question(interaction: Dict[str, str]) -> Dict[str, str]`:
    - *Purpose*: Generates adaptive follow-up questions adapted to prior user responses.
    - *Where called*: In `questioner_pipeline.py` when `count > 0` (Turns 2 to 5, exactly 4 times).
    - *LangChain Components*: **`HumanMessage` & `AIMessage`**
      - *Where used*: Appends `HumanMessage(content=user_a)` to record the user's prior answer, then appends `AIMessage(content=next_question_text)` to record the newly generated question.
      - *Why used*: Maintains alternating dialogue turn sequence in `self.memory` passed to `MessagesPlaceholder`.

  - `get_qa_history() -> Dict[str, str]`:
    - *Purpose*: Returns the final mapping of questions to user answers.
    - *Where called*: At the end of `run()` in `questioner_pipeline.py`.

---

#### C. `questioner_pipeline.py`
**Purpose**: Pipeline execution entry point managing user CLI interaction and file output.

- **Functions**:
  - `run(max_turns=5, output_filepath="qa_transcript.json") -> Dict[str, str]`:
    - *Purpose*: Orchestrates the turn-based loop, controls which agent function is called based on `count`, captures user inputs, and saves results.
    - *Where called*: When running `python questioner_pipeline.py` or imported by orchestrator workflows.
    - *LangChain Component*: **`HumanMessage`**
      - *Where used*: Records the 5th (final) user answer into `questioner.memory` after the loop exits.
      - *Why used*: Ensures the complete 5-turn session is fully recorded in memory before exporting.

---

## AGENT ORCHESTRATION

### 1. Execution Workflow
The Agent Orchestrator acts as the centralized coordinator that partitions the Q&A transcript produced by the Questioner Agent into specialized subsets for downstream domain specialist agents (`reasoning_agent` and `cognitive_agent`):

1. **Pipeline Invocation (`run` in `ao_pipeline.py`)**:
   - `run(transcript_path="qa_transcript.json", output_filepath="routed_tasks.json")` is called.
   - Accepts either a string file path or a pre-loaded dictionary of Q&A pairs.
   - If a file path is provided, it deserializes the JSON transcript containing `{ question: user_answer }` pairs.

2. **Orchestrator Agent Instantiation**:
   - `OrchestratorAgent(llm=...)` is initialized:
     - Configures the structured output parser `PydanticOutputParser(pydantic_object=OrchestratorOutput)`.
     - Calls `_build_chain()` to compile the LCEL classification chain: `PromptTemplate | LLM | PydanticOutputParser`.
     - Calls `_build_graph()` to construct and compile the LangGraph workflow: `START -> "classify" -> END`.

3. **Orchestration Execution (`orchestrator.orchestrate(qa_data)`)**:
   - `orchestrate(raw_qa)` invokes the compiled LangGraph workflow: `self.graph.invoke({"raw_qa": raw_qa})`.
   - Execution begins at `START` and transitions directly to the `"classify"` node.

4. **Node Execution (`_classify_node`)**:
   - Extracts `raw_qa` from state.
   - If empty, immediately returns empty task sets with a zeroed summary.
   - Invokes `self.chain.invoke({"qa_json": json.dumps(raw_qa, indent=2)})`.
   - The LLM partitions each Q&A pair into either `reasoning_agent` or `cognitive_agent` based on cognitive evaluation criteria.
   - Directly tallies and computes the task distribution summary:
     - `total_tasks`: Total routed Q&A pairs.
     - `reasoning_tasks_count`: Number of tasks assigned to Reasoning Agent.
     - `cognitive_tasks_count`: Number of tasks assigned to Cognitive Agent.
   - Returns `reasoning_agent`, `cognitive_agent`, and `summary` into the state.
   - Transitions directly from `"classify"` to `END`.

5. **Persistence & Handoff (`run` in `ao_pipeline.py`)**:
   - Extracts `reasoning_agent` and `cognitive_agent` mappings from the graph output.
   - Writes the partitioned tasks directly to `routed_tasks.json` as the input bridge for `ReasoningAgent` and `CognitiveAgent`.
   - Returns the complete result dictionary.

```
[run() entry] ──► Load "qa_transcript.json"
       │
       ▼
[OrchestratorAgent.orchestrate()]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ LangGraph State Machine (OrchestratorState)            │
│                                                        │
│  START ──► [classify] ──► END                          │
│                 │                                      │
│                 ├─► LCEL Chain Invocation              │
│                 │   (PromptTemplate | LLM | Parser)    │
│                 │                                      │
│                 └─► Partition & Compute Summary:       │
│                     - reasoning_agent: Dict[str, str]  │
│                     - cognitive_agent: Dict[str, str]  │
│                     - summary: Task Counts             │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
         Export partitions to "routed_tasks.json"
                          │
                          ▼
                    Return Result
```

---

### 2. File & Component Breakdown (with LangChain & LangGraph Components)

#### A. `orchestrator.py`
**Purpose**: Houses the output schemas, LCEL classification chain, and compiled LangGraph state graph for Q&A task categorization and routing.

- **Classes & Schemas**:
  - `OrchestratorOutput(BaseModel)`:
    - *Purpose*: Pydantic schema enforcing structured allocation of Q&A pairs into `reasoning_agent` and `cognitive_agent` dictionaries.
    - *LangChain Component*: **`PydanticOutputParser`**
      - *Where used*: `self.parser = PydanticOutputParser(pydantic_object=OrchestratorOutput)`, chained into `self.chain`.
      - *Why used*: Guarantees that the LLM returns strict, valid JSON conforming to `{ "reasoning_agent": {...}, "cognitive_agent": {...} }`.
  - `OrchestratorState(TypedDict)`:
    - *Purpose*: State schema tracking `raw_qa`, `reasoning_agent`, `cognitive_agent`, and `summary`.
    - *LangGraph Component*: **`StateGraph(OrchestratorState)`**
      - *Where used*: Passed to `StateGraph` in `_build_graph()`.
      - *Why used*: Defines the strongly typed state schema for channels propagated across LangGraph execution steps.

- **Methods**:
  - `__init__(llm=None)`:
    - *Purpose*: Configures LLM, output parser, builds the LCEL chain, and compiles the LangGraph workflow.
    - *Where called*: By `run()` in `ao_pipeline.py`.

  - `_build_chain()`:
    - *Purpose*: Assembles the prompt template and binds it to the LLM and output parser.
    - *Where called*: Automatically inside `__init__`.
    - *LangChain Components Used*:
      - **`PromptTemplate`** (`langchain_core.prompts`):
        - *Where used*: Formulates the prompt with `qa_json` and `format_instructions`.
        - *Why used*: Injects the raw Q&A JSON transcript and formatting constraints cleanly into the prompt context.
      - **LCEL Pipe Operator (`|`)**:
        - *Where used*: `self.chain = prompt | self.llm | self.parser`.
        - *Why used*: Creates an atomic runnable pipeline that takes input variables, executes model inference, and parses the output into a validated Pydantic model.

  - `_classify_node(state: OrchestratorState) -> Dict[str, Any]`:
    - *Purpose*: Consolidated graph node that executes classification via `self.chain.invoke(...)`, counts assigned tasks, and computes summary statistics.
    - *Where called*: Executed by the LangGraph runtime for the `"classify"` node.
    - *LangGraph Component*: **Graph Node Function (`workflow.add_node`)**
      - *Where used*: Bound via `workflow.add_node("classify", self._classify_node)`.
      - *Why used*: Encapsulates LLM partitioning and count tallying within a single deterministic graph node.

  - `_build_graph()`:
    - *Purpose*: Constructs and compiles the LangGraph state machine.
    - *Where called*: Automatically inside `__init__`.
    - *LangGraph Components Used*:
      - **`StateGraph`** (`langgraph.graph`):
        - *Where used*: `workflow = StateGraph(OrchestratorState)`.
        - *Why used*: Provides the graph-based state orchestration engine to manage data flow.
      - **`START` & `END`** (`langgraph.graph`):
        - *Where used*: `workflow.add_edge(START, "classify")` and `workflow.add_edge("classify", END)`.
        - *Why used*: LangGraph sentinel nodes that mark workflow entry and completion.
      - **`workflow.compile()`**:
        - *Where used*: `self.graph = workflow.compile()`.
        - *Why used*: Compiles the state graph definition into an invocable `CompiledStateGraph` runnable.

  - `orchestrate(raw_qa: Dict[str, str]) -> Dict[str, Any]`:
    - *Purpose*: Public API method executing the compiled state machine with input Q&A transcript.
    - *Where called*: By `run()` in `ao_pipeline.py`.

---

#### B. `ao_pipeline.py`
**Purpose**: Pipeline execution entry point coordinating transcript loading, orchestrator execution, and task file export.

- **Functions**:
  - `run(transcript_path="qa_transcript.json", output_filepath="routed_tasks.json") -> Dict[str, Any]`:
    - *Purpose*: Single consolidated function running orchestrator operations in orderly sequence: loading input, instantiating `OrchestratorAgent`, invoking `orchestrate()`, saving partitions to `routed_tasks.json`, and returning the final state.
    - *Where called*: When running `python ao_pipeline.py` or imported as the orchestration stage in higher-level pipelines.

---

## REASONING AGENT

### 1. Execution Workflow
The Reasoning Agent evaluates the user's deductive/inductive logic, problem-solving strategies, critical thinking, and premise validation from the tasks routed to it by the Agent Orchestrator:

1. **Pipeline Invocation (`run_reasoning_pipeline` / `run` in `reasoning_pipeline.py`)**:
   - `run(source="routed_tasks.json", output_filepath="agents_analysis.json")` is called.
   - Accepts either a file path to `routed_tasks.json` or a pre-loaded dictionary of routed tasks.

2. **Task Ingestion (`load_reasoning_input`)**:
   - Reads `routed_tasks.json` and extracts the subset of Q&A pairs allocated under `"reasoning_agent"`.
   - Returns the extracted tasks as a clean dictionary `{ question: answer }`.

3. **Reasoning Agent Instantiation**:
   - `ReasoningAgent(llm=...)` is initialized:
     - Configures the output parser `PydanticOutputParser(pydantic_object=ReasoningAnalysisOutput)`.
     - Injects parser format instructions into `PromptTemplate(input_variables=["tasks"])`.
     - Compiles the atomic LCEL runnable chain: `self.prompt | self.llm | self.parser`.

4. **Reasoning Evaluation (`analyze_reasoning`)**:
   - `reasoning_agent.analyze_reasoning(tasks)` is invoked with the extracted task dictionary.
   - Formats tasks as JSON and executes the LCEL chain: `self.chain.invoke({"tasks": json.dumps(tasks, indent=2)})`.
   - The LLM evaluates logical reasoning quality, problem-solving approach, error patterns, and overall reasoning assessment.
   - Output parser returns validated `ReasoningAnalysisOutput`.
   - Method returns a structured dictionary: `{"qna": parsed.qna, "reasoning_result": parsed.reasoning_result}`.

5. **Persistence & Handoff (`save_agents_analysis` / `create_agents_analysis_json`)**:
   - Checks if `agents_analysis.json` already exists; if so, reads and preserves existing data.
   - Updates `output["reasoning_agent"] = analysis_data`.
   - Writes the updated JSON object to `agents_analysis.json`.
   - Returns the analysis result.

```
[run() entry] ──► Read "routed_tasks.json"
       │
       ▼
[load_reasoning_input()] ──► Extract "reasoning_agent" dict
       │
       ▼
[ReasoningAgent.analyze_reasoning(tasks)]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ LCEL Chain (ReasoningAnalysisOutput)                   │
│                                                        │
│  tasks (JSON) ──► PromptTemplate ──► LLM ──► Parser    │
│                                                        │
│  Returns:                                              │
│    - qna: Dict[str, Any]                               │
│    - reasoning_result: str                             │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
[save_agents_analysis()] ──► Update output["reasoning_agent"]
                          │
                          ▼
            Save to "agents_analysis.json"
```

---

### 2. File & Component Breakdown (with LangChain Components)

#### A. `reason_agent.py`
**Purpose**: Houses the output schema, prompt compilation, and LCEL chain for logical reasoning evaluation.

- **Classes & Schemas**:
  - `ReasoningAnalysisOutput(BaseModel)`:
    - *Purpose*: Pydantic schema enforcing structured output containing `qna: Dict[str, Any]` and `reasoning_result: str`.
    - *LangChain Component*: **`PydanticOutputParser`**
      - *Where used*: `self.parser = PydanticOutputParser(pydantic_object=ReasoningAnalysisOutput)`, chained into `self.chain`.
      - *Why used*: Guarantees that the LLM returns strict, valid JSON conforming to `{ "qna": {...}, "reasoning_result": "..." }`.

- **Methods**:
  - `__init__(llm=None)`:
    - *Purpose*: Configures LLM, parser, binds format instructions, and builds the LCEL chain.
    - *Where called*: By `run_reasoning_pipeline()` in `reasoning_pipeline.py`.
    - *LangChain Components Used*:
      - **`PromptTemplate`** (`langchain_core.prompts`):
        - *Where used*: `self.prompt = PromptTemplate(...)` with `input_variables=["tasks"]` and `format_instructions`.
        - *Why used*: Injects the assigned reasoning tasks and Pydantic format instructions into an explicit evaluation prompt.
      - **LCEL Pipe Operator (`|`)**:
        - *Where used*: `self.chain = self.prompt | self.llm | self.parser`.
        - *Why used*: Connects prompt formatting, model inference, and output parsing into an atomic executable runnable pipeline.

  - `analyze_reasoning(tasks: Dict[str, str]) -> Dict[str, Any]`:
    - *Purpose*: Invokes the LCEL chain on the assigned tasks and returns `{ "qna": parsed.qna, "reasoning_result": parsed.reasoning_result }`.
    - *Where called*: In `reasoning_pipeline.py`.

---

#### B. `reasoning_pipeline.py`
**Purpose**: Pipeline execution entry point coordinating task ingestion from `routed_tasks.json`, reasoning analysis, and persistence to `agents_analysis.json`.

- **Functions**:
  - `load_reasoning_input(source="routed_tasks.json") -> Dict[str, str]`:
    - *Purpose*: Deserializes `routed_tasks.json` and extracts the dictionary of tasks allocated to `"reasoning_agent"`.
    - *Where called*: By `run_reasoning_pipeline()`.
  - `save_agents_analysis(analysis_data, output_filepath="agents_analysis.json") -> None` (aliased as `create_agents_analysis_json`):
    - *Purpose*: Persists the analysis to `agents_analysis.json` under `"reasoning_agent"` while safely preserving evaluations from other agents.
    - *Where called*: By `run_reasoning_pipeline()`.
  - `run_reasoning_pipeline(source="routed_tasks.json", output_filepath="agents_analysis.json") -> Dict[str, Any]` (aliased as `run`):
    - *Purpose*: Orchestrates the complete reasoning workflow in orderly sequence: loading input tasks, running `ReasoningAgent`, saving to `agents_analysis.json`, and returning the analysis dictionary.
    - *Where called*: When running `python reasoning_pipeline.py` or imported by orchestrator workflows.

---

## COGNITIVE AGENT

### 1. Execution Workflow
The Cognitive Agent evaluates the user's pattern recognition ability, numerical/spatial sequence reasoning, working memory retention & tracking, and attention to detail from the tasks routed to it by the Agent Orchestrator:

1. **Pipeline Invocation (`run_cognitive_pipeline` / `run` in `cognitive_pipeline.py`)**:
   - `run(source="routed_tasks.json", output_filepath="agents_analysis.json")` is called.
   - Accepts either a file path to `routed_tasks.json` or a pre-loaded dictionary of routed tasks.

2. **Task Ingestion (`load_cognitive_input`)**:
   - Reads `routed_tasks.json` and extracts the subset of Q&A pairs allocated under `"cognitive_agent"`.
   - Returns the extracted tasks as a clean dictionary `{ question: answer }`.

3. **Cognitive Agent Instantiation**:
   - `CognitiveAgent(llm=...)` is initialized:
     - Configures the output parser `PydanticOutputParser(pydantic_object=CognitiveAnalysisOutput)`.
     - Injects parser format instructions into `PromptTemplate(input_variables=["tasks"])`.
     - Compiles the atomic LCEL runnable chain: `self.prompt | self.llm | self.parser`.

4. **Cognitive Evaluation (`analyze_cognitive`)**:
   - `cognitive_agent.analyze_cognitive(tasks)` is invoked with the extracted task dictionary.
   - Formats tasks as JSON and executes the LCEL chain: `self.chain.invoke({"tasks": json.dumps(tasks, indent=2)})`.
   - The LLM evaluates pattern recognition ability, working memory retention and tracking, attention to detail, and overall cognitive assessment.
   - Output parser returns validated `CognitiveAnalysisOutput`.
   - Method returns a structured dictionary: `{"qna": parsed.qna, "cognitive_result": parsed.cognitive_result}`.

5. **Persistence & Handoff (`save_agents_analysis` / `create_agents_analysis_json`)**:
   - Checks if `agents_analysis.json` already exists; if so, reads and preserves existing data (including `reasoning_agent`).
   - Updates `output["cognitive_agent"] = analysis_data`.
   - Writes the updated JSON object to `agents_analysis.json`.
   - Returns the analysis result.

```
[run() entry] ──► Read "routed_tasks.json"
       │
       ▼
[load_cognitive_input()] ──► Extract "cognitive_agent" dict
       │
       ▼
[CognitiveAgent.analyze_cognitive(tasks)]
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ LCEL Chain (CognitiveAnalysisOutput)                   │
│                                                        │
│  tasks (JSON) ──► PromptTemplate ──► LLM ──► Parser    │
│                                                        │
│  Returns:                                              │
│    - qna: Dict[str, Any]                               │
│    - cognitive_result: str                             │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
[save_agents_analysis()] ──► Update output["cognitive_agent"]
                          │
                          ▼
            Save to "agents_analysis.json"
```

---

### 2. File & Component Breakdown (with LangChain Components)

#### A. `cognitive_agent.py`
**Purpose**: Houses the output schema, prompt compilation, and LCEL chain for cognitive capability evaluation.

- **Classes & Schemas**:
  - `CognitiveAnalysisOutput(BaseModel)`:
    - *Purpose*: Pydantic schema enforcing structured output containing `qna: Dict[str, Any]` and `cognitive_result: str`.
    - *LangChain Component*: **`PydanticOutputParser`**
      - *Where used*: `self.parser = PydanticOutputParser(pydantic_object=CognitiveAnalysisOutput)`, chained into `self.chain`.
      - *Why used*: Guarantees that the LLM returns strict, valid JSON conforming to `{ "qna": {...}, "cognitive_result": "..." }`.

- **Methods**:
  - `__init__(llm=None)`:
    - *Purpose*: Configures LLM, parser, binds format instructions, and builds the LCEL chain.
    - *Where called*: By `run_cognitive_pipeline()` in `cognitive_pipeline.py`.
    - *LangChain Components Used*:
      - **`PromptTemplate`** (`langchain_core.prompts`):
        - *Where used*: `self.prompt = PromptTemplate(...)` with `input_variables=["tasks"]` and `format_instructions`.
        - *Why used*: Injects the assigned cognitive tasks and Pydantic format instructions into an explicit evaluation prompt.
      - **LCEL Pipe Operator (`|`)**:
        - *Where used*: `self.chain = self.prompt | self.llm | self.parser`.
        - *Why used*: Connects prompt formatting, model inference, and output parsing into an atomic executable runnable pipeline.

  - `analyze_cognitive(tasks: Dict[str, str]) -> Dict[str, Any]`:
    - *Purpose*: Invokes the LCEL chain on the assigned tasks and returns `{ "qna": parsed.qna, "cognitive_result": parsed.cognitive_result }`.
    - *Where called*: In `cognitive_pipeline.py`.

---

#### B. `cognitive_pipeline.py`
**Purpose**: Pipeline execution entry point coordinating task ingestion from `routed_tasks.json`, cognitive analysis, and persistence to `agents_analysis.json`.

- **Functions**:
  - `load_cognitive_input(source="routed_tasks.json") -> Dict[str, str]`:
    - *Purpose*: Deserializes `routed_tasks.json` and extracts the dictionary of tasks allocated to `"cognitive_agent"`.
    - *Where called*: By `run_cognitive_pipeline()`.
  - `save_agents_analysis(analysis_data, output_filepath="agents_analysis.json") -> None` (aliased as `create_agents_analysis_json`):
    - *Purpose*: Persists the analysis to `agents_analysis.json` under `"cognitive_agent"` while safely preserving evaluations from other agents.
    - *Where called*: By `run_cognitive_pipeline()`.
  - `run_cognitive_pipeline(source="routed_tasks.json", output_filepath="agents_analysis.json") -> Dict[str, Any]` (aliased as `run`):
    - *Purpose*: Orchestrates the complete cognitive workflow in orderly sequence: loading input tasks, running `CognitiveAgent`, saving to `agents_analysis.json`, and returning the analysis dictionary.
    - *Where called*: When running `python cognitive_pipeline.py` or imported by orchestrator workflows.

---

## FINAL REPORT AGENT

### 1. Execution Workflow
The Final Report Agent synthesizes the evaluations from both specialist agents (`reasoning_agent` and `cognitive_agent`) into a unified, comprehensive cognitive profile. The execution flow in `report_pipeline.py` executes in a strictly orderly sequence:

1. **Pipeline Entry (`run_report_pipeline` / `run` in `report_pipeline.py`)**:
   - `run(source="agents_analysis.json", output_filepath="final_report.md")` is invoked.
   - Accepts either a string path to `agents_analysis.json` or an in-memory dictionary.

2. **Step 1: Ingestion (`load_analysis_input`)**:
   - `analysis_data = load_analysis_input(source)`
   - Deserializes and returns the complete specialist analyses dictionary containing evaluations from both `reasoning_agent` and `cognitive_agent`.

3. **Step 2: Agent Instantiation (`ReportAgent`)**:
   - `report_agent = ReportAgent()`
   - Initializes the LLM (`ChatHuggingFace`), the output parser (`StrOutputParser`), and formats `PromptTemplate(input_variables=["analysis_json"])`.
   - Compiles the atomic LCEL chain: `self.chain = self.prompt | self.llm | self.parser`.

4. **Step 3: Synthesis & Report Generation (`generate_report`)**:
   - `report = report_agent.generate_report(analysis_data)`
   - Serializes `analysis_data` into formatted JSON: `json.dumps(analysis_data, indent=2)`.
   - Invokes `self.chain.invoke({"analysis_json": ...})`.
   - The LLM synthesizes disparate cognitive dimensions, balances strengths against blind spots, and produces a structured continuous Markdown report string parsed directly by `StrOutputParser`.

5. **Step 4: Persistence (`save_report`)**:
   - `save_report(report, output_filepath)`
   - Writes the formatted report text to `final_report.md`.

6. **Step 5: Return & Output Presentation**:
   - `return report` returns the string to the caller.
   - When executed as a script (`if __name__ == "__main__":`), `result = run()` executes and `print(result)` outputs the final report to stdout.

```
[run() entry] ──► source = "agents_analysis.json"
       │
       ▼
 1. analysis_data = load_analysis_input(source)
       │
       ▼
 2. report_agent = ReportAgent()
       │
       ▼
 3. report = report_agent.generate_report(analysis_data)
       │
       ├─► Serialization: json.dumps(analysis_data, indent=2)
       ├─► LCEL Chain: prompt | llm | parser (StrOutputParser)
       └─► Return: raw Markdown report string
       │
       ▼
 4. save_report(report, output_filepath="final_report.md")
       │
       ▼
 5. return report ──► Printed to stdout via if __name__ == "__main__"
```

---

### 2. File & Component Breakdown (with LangChain Components)

#### A. `report_agent.py`
**Purpose**: Houses the synthesis prompt and LCEL chain producing the unified user cognitive profile.

- **Classes**:
  - `ReportAgent`:
    - `__init__(llm=None)`:
      - *Purpose*: Configures LLM, prompt template, output parser, and compiles the LCEL chain.
      - *Where called*: By `run_report_pipeline()` in `report_pipeline.py`.
      - *LangChain Components Used*:
        - **`StrOutputParser`** (`langchain_core.output_parsers`):
          - *Where used*: `self.parser = StrOutputParser()`, chained into `self.chain`.
          - *Why used*: Streams the LLM's formatted Markdown text directly as a plain string without Pydantic schema validation overhead.
        - **`PromptTemplate`** (`langchain_core.prompts`):
          - *Where used*: `self.prompt = PromptTemplate(...)` with `input_variables=["analysis_json"]`.
          - *Why used*: Structures synthesis guidelines covering persona, strengths, blind spots, and development recommendations.
        - **LCEL Pipe Operator (`|`)**:
          - *Where used*: `self.chain = self.prompt | self.llm | self.parser`.
          - *Why used*: Creates an atomic runnable executing prompt formatting, model inference, and string parsing in a single step.
    - `generate_report(analysis_data: Dict[str, Any]) -> str`:
      - *Purpose*: Invokes the LCEL chain with serialized analysis data and returns the final report string.
      - *Where called*: In `report_pipeline.py`.

---

#### B. `report_pipeline.py`
**Purpose**: Pipeline execution entry point coordinating analysis loading, report generation, and Markdown file persistence.

- **Functions**:
  - `load_analysis_input(source="agents_analysis.json") -> Dict[str, Any]`:
    - *Purpose*: Reads and deserializes specialist evaluations from `agents_analysis.json` or returns the passed dictionary.
    - *Where called*: Step 1 in `run_report_pipeline()`.
  - `save_report(report_text: str, output_filepath="final_report.md") -> None`:
    - *Purpose*: Writes the generated report text directly to `final_report.md`.
    - *Where called*: Step 4 in `run_report_pipeline()`.
  - `run_report_pipeline(source="agents_analysis.json", output_filepath="final_report.md") -> str` (aliased as `run`):
    - *Purpose*: Main execution entry point coordinating analysis loading, report generation, file persistence, and return.
    - *Where called*: When running `python report_pipeline.py` or imported as the final stage of the multi-agent pipeline.
  - `if __name__ == "__main__":`:
    - *Purpose*: Runs `result = run()` and prints the full Markdown report to stdout.

---

# COMPLETE END-TO-END SYSTEM WORKFLOW

The multi-agent system operates across four coordinated stages through standardized file bridges:

```
                      [User Turn-Based Interaction]
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │      QUESTIONER AGENT       │
                     │  (5-turn dynamic dialogue)  │
                     └──────────────┬──────────────┘
                                    │ writes
                                    ▼
                          qa_transcript.json
                                    │
                                    ▼ reads
                     ┌─────────────────────────────┐
                     │     AGENT ORCHESTRATOR      │
                     │   (LangGraph partitioner)   │
                     └──────────────┬──────────────┘
                                    │ writes
                                    ▼
                           routed_tasks.json
                                    │
                  ┌─────────────────┴─────────────────┐
                  │ reads                             │ reads
                  ▼                                   ▼
   ┌─────────────────────────────┐     ┌─────────────────────────────┐
   │       REASONING AGENT       │     │       COGNITIVE AGENT       │
   │  (Deductive & Logic Rubric) │     │ (Pattern & Attention Rubric)│
   └──────────────┬──────────────┘     └──────────────┬──────────────┘
                  │ updates                           │ updates
                  └─────────────────┬─────────────────┘
                                    │ writes
                                    ▼
                          agents_analysis.json
                                    │
                                    ▼ reads
                     ┌─────────────────────────────┐
                     │     FINAL REPORT AGENT      │
                     │  (Synthesis & Persona LCEL) │
                     └──────────────┬──────────────┘
                                    │ writes
                                    ▼
                           final_report.md
```



