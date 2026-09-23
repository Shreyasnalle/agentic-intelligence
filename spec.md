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
