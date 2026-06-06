# ARA: Autonomous Financial Research Agent

The ARA is an advanced Agentic AI system designed to replicate the end-to-end research workflow of a professional financial analyst. By leveraging a **Plan-and-Execute reasoning loop**, the agent autonomously decomposes complex research queries, interacts with a diverse tool registry, and synthesizes high-fidelity financial reports.

## Core Architecture
* **Reasoning Loop:** Implements a Plan-and-Execute paradigm for complex query decomposition.
* **Tool Registry:** Features 10+ specialized tools for SEC EDGAR retrieval, financial APIs, earnings call analysis, and global news synthesis.
* **Tri-Layer Memory:** * *Short-term:* Session-based context management via LangGraph.
    * *Long-term:* Vector database integration for historical financial data retrieval.
    * *Episodic:* Captures research paths to improve performance across multi-step tasks.
* **Conflict Resolution:** A multi-source synthesis engine designed to resolve data discrepancies across conflicting financial reporting sources.

## Performance Validation
The agent was rigorously tested against the Zetheta benchmark framework, successfully navigating 8 progressive challenges of increasing difficulty.

| Challenge Tier | Result | Focus |
| :--- | :--- | :--- |
| **Basic** | 5/5 | SEC Filing & Fundamental Metric Accuracy |
| **Intermediate** | 5/5 | Source Hierarchy & Sentiment Analysis |
| **Stress** | 5/5 | Complex Calculation & Constraint Adherence |

*Detailed execution traces and validation logs are available in [TRACES.md](TRACES.md).*

## Tech Stack
* **Framework:** LangGraph (for stateful multi-actor orchestration)
* **Reasoning Engine:** [Insert your model name, e.g., GPT-4o or Claude 3.5 Sonnet]
* **Orchestration:** Plan-and-Execute Agentic Loop
* **Evaluation:** Custom 20-metric automated benchmarking suite

## Setup Instructions

### 1. Requirements
* Python 3.10+
* API Keys for: [e.g., OpenAI, Tavily, SEC EDGAR access]

### 2. Installation
```bash
git clone [your-repo-link]
cd ARA_Financial_Agent
pip install -r requirements.txt
```

### 3. Execution
Run the evaluation suite to validate the agent against the project benchmarks:

```bash
python evaluation/evaluator.py
```

## Resilience & Error Handling
The agent incorporates Fallback Chains at every tool interaction point. If an API call fails or results are inconsistent, the agent automatically switches to alternative retrieval strategies (e.g., web-fallback to SEC-EDGAR direct access) to ensure zero human intervention is required.

Project developed under the Zetheta Agentic AI Engineering methodology.