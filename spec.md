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

