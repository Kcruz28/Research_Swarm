# Research_Swarm

A multi-agent research paper analysis system using LangChain, LangGraph, and Ollama LLM.

## Overview

Research_Swarm is an orchestrated workflow that analyzes research papers through multiple specialized agents:
- **Analyst**: Creates detailed summaries of research papers
- **Critic**: Provides critique of the analysis
- **Refiner**: Refines the summary based on feedback

The agents work together in a sequential pipeline using LangGraph's StateGraph.

## Requirements

- **Python**: 3.12 or higher
- **Ollama**: Local LLM runtime (for running language models locally)

## Installation

### 1. Prerequisites
Ensure you have Python 3.12+ installed:
```bash
python3 --version
```

Install Ollama from: https://ollama.ai

### 2. Create Virtual Environment
```bash
python3.12 -m venv research_env
source research_env/bin/activate  # On Windows: research_env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Ollama Model
Pull a language model (qwen2.5-coder is used by default):
```bash
ollama pull qwen2.5-coder:1.5b
```

Start Ollama service:
```bash
ollama serve
```

## Usage

### Running the Project
```bash
python3 main.py
```

### Expected Output
The script will:
1. Create an Orchestration with 3 agent nodes (analysist, critic, refiner)
2. Connect them in a pipeline: START → analysist → critic → refiner → END
3. Invoke the graph with a user message
4. Display all agent responses

Example:
```
============================================================
GRAPH EXECUTION RESULTS
============================================================

Message 0: USER
Content: Analyze this research paper...

Message 1: AI
Content: Analyst: [detailed summary]...

Message 2: AI
Content: Critic: [critique]...

Message 3: AI
Content: Refiner: [refined summary]...
```

## Project Structure
```
Research_Swarm/
├── main.py              # Entry point
├── orchestration.py     # Graph orchestration and workflow logic
├── agents.py           # Agent definitions (Analyst, Critic, Refiner)
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## How It Works

### State Flow
Each node receives and updates `MessagesState`:
```python
START
  ↓
Input message
  ↓
analysist (analyzes research paper)
  ↓
critic (critiques the analysis)
  ↓
refiner (refines based on criticism)
  ↓
END
  ↓
Output: All accumulated messages
```

### Key Concepts
- **StateGraph**: Defines the workflow structure
- **MessagesState**: State schema containing message history
- **Nodes**: Agent functions that process state
- **Edges**: Connections defining execution flow
- **invoke()**: Executes the entire graph pipeline

## Troubleshooting

### Issue: `ModuleNotFoundError`
- Ensure virtual environment is activated: `source research_env/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Issue: Connection refused to Ollama
- Verify Ollama is running: `ollama serve`
- Check Ollama client URL in `agents.py` (default: `http://localhost:11434`)

### Issue: Model not found
- List available models: `ollama ls`
- Pull the required model: `ollama pull qwen2.5-coder:1.5b`

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| langchain-ollama | ≥0.1.0 | Ollama LLM integration |
| langgraph | ≥0.0.1 | Graph orchestration framework |
| langchain | ≥0.1.0 | LLM abstraction and tools |
| python-dotenv | ≥1.0.0 | Environment variable management |
