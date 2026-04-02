# Research_Swarm

Research_Swarm is a small multi-agent pipeline for reading a research PDF, drafting a summary, critiquing it, and producing a refined final version using LangGraph and a local Ollama model.

## What It Does

The workflow is split across three agents:

- `analysist`: reads the paper and creates an initial technical summary
- `critic`: checks the summary against the source text and rejects weak coverage
- `refiner`: produces the final polished summary after review

The app also prints a trace table of the conversation and renders the final refined summary with Rich Markdown.

## Current Structure

```text
Research_Swarm/
├── main.py
├── orchestration.py
├── agents.py
├── pdf_reader.py
├── requirements.txt
├── test.py
├── test_agents.py
└── README.md
```

## Requirements

- Python 3.12+
- Ollama running locally at `http://localhost:11434`
- A pulled model that matches `agents.py`, currently `qwen3.5:4b`

## Setup

1. Create and activate the virtual environment:

```bash
python3.12 -m venv research_env
source research_env/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Start Ollama and pull the configured model:

```bash
ollama serve
ollama pull qwen3.5:4b
```

## Run

The default entry point is `main.py`:

```bash
python main.py
```

By default, the app loads the PDF named `IEEE Xplore Full-Text PDF_.pdf` from the project root. If your file has a different name or location, update `FILE_PATH` in [main.py](main.py).

## Data Flow

1. `PDFReader` loads the PDF through Docling and OCR-aware pipeline options.
2. `Orchestration` builds a LangGraph `MessagesState` workflow.
3. `Agents` runs the analyst, critic, and refiner nodes with `ChatOllama`.
4. `main.py` prints the message history and the final refined summary.

The graph loop is intentionally simple: `START -> analysist -> critic -> (analysist or refiner) -> END`.

## Dependencies

The project currently uses these packages:

- `langchain-ollama`
- `langgraph`
- `langchain`
- `langchain-core`
- `langchain-community`
- `langchain-docling`
- `docling`
- `rich`
- `pydantic`
- `python-dotenv`

## Troubleshooting

- If you see `NameError: HumanMessage is not defined`, make sure `langchain-core` is installed and that [main.py](main.py) and [agents.py](agents.py) import it correctly.
- If Ollama connection fails, confirm `ollama serve` is running and the base URL in [agents.py](agents.py) is still `http://localhost:11434`.
- If the PDF does not load, verify the file path in [pdf_reader.py](pdf_reader.py) and ensure the document is readable by Docling.

## Notes

- The method name `analysist` is intentionally kept as-is because it is referenced throughout the graph wiring.
- The project includes `test.py` and `test_agents.py`, but the main runtime path is `main.py`.
