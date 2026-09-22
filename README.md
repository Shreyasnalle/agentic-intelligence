# Agentic Intelligence

## 1. Project Overview & Objective

The goal of this project is to build an autonomous multi-agent AI system that evaluates a user's cognitive and analytical capabilities through dynamic interaction. Instead of using a standard single prompt-to-response chain, the system coordinates multiple specialized agents that interact, evaluate specific intellectual facets, and synthesize their findings into a unified, structured intelligence profile.

---

## 2. Step-by-Step Execution Flow

* **Step 1: Adaptive Interview (Counselor Agent)**
  * Dynamically generates and presents evaluation questions to the user.
  * Adapts subsequent prompts in real time based on the user's prior answers to test limits, adjust difficulty, or probe different cognitive angles.
  * Enforces a termination threshold (e.g., after 3–4 targeted interactions) and serializes the complete dialogue session into a structured JSON transcript.

* **Step 2: Parsing & Dynamic Routing (Agent Orchestrator)**
  * Ingests the raw Q&A JSON data from the Counselor Agent.
  * Dynamically assesses the nature of each prompt and response rather than relying on static routing.
  * Distributes the workload by delegating logic/deduction problems to the Reasoning Analyst and pattern/working-memory tasks to the Cognitive Analyst.

* **Step 3: Specialized Multi-Agent Analysis**
  * **Reasoning Analyst**: Evaluates deductive and inductive logic, critical thinking, problem-solving heuristics, and premise validation.
  * **Cognitive Analyst**: Analyzes pattern recognition, attention to operational details, working memory, and sequential reasoning.
  * Both specialists execute their evaluations in parallel and return structured JSON records adhering to strict evaluation rubrics.

* **Step 4: Synthesis & Reporting (Report Agent / Profile Analyst)**
  * Collects and joins the outputs from both specialist agents in a fan-in pattern.
  * Reconciles disparate or contradictory findings across cognitive domains.
  * Generates a coherent, comprehensive final intelligence profile summarizing performance metrics, cognitive strengths, potential blind spots, and an overall profile persona.

---

## Architecture & Workflow Diagram

![Agent Workflow](assets/workflow.png)

---

## Core Technical Deliverables

* **Orchestration Framework**: Implement the workflow using a stateful graph architecture (such as LangGraph) to handle dynamic agent routing, state cycles, and parallel execution.
* **State Management**: Maintain a centralized, strictly-typed session state accessible by all participating nodes.
* **Data Schemas & Validation**: Define explicit Pydantic schemas for inter-agent messages and outputs (such as Q&A payloads, specialist evaluation metrics, and final report structures) to eliminate hallucinated schemas and malformed responses.
* **Defensive Engineering & Error Handling**: Implement retries, JSON parsing fallback routines, and error-recovery handlers to gracefully manage API rate limits or invalid model responses.

---

## Assessment Deliverables & Submission Checklist

* [ ] **Python Codebase**: Clean, modular repository implementing all agents, routing, and state workflows.
* [ ] **Dynamic Routing**: Autonomous orchestration deciding agent delegation without hardcoded static rules.
* [ ] **README Documentation**: Comprehensive setup instructions, environment configurations, and an architectural diagram.
* [ ] **Demo Video**: A 3–5 minute recorded walkthrough demonstrating end-to-end execution and explaining key architectural choices.
