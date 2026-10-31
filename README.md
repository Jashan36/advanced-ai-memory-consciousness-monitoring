# Advanced AI Memory and Consciousness Monitoring System

A Python research prototype combining persistent memory, knowledge retrieval, behavior monitoring, visualization, and optional parameter-efficient model adaptation.

> **Important:** “Consciousness” metrics in this project are heuristic indicators of system behavior. They do not measure or establish subjective experience, sentience, or consciousness.

## Features

- Memory storage and retrieval with configurable capacity and retention.
- External knowledge storage and similarity-based retrieval.
- Heuristic monitoring of attention, consistency, context use, and other behavior signals.
- Multi-threaded memory processing.
- Optional LoRA-style model adaptation.
- Visualization tools for system metrics.

Optional integrations (such as Transformers, FAISS, and scikit-learn) are used when installed; some functionality may be limited without them.

## Requirements

- Python 3.10 or newer
- PyTorch and NumPy for the core system
- Pandas, Matplotlib, and Seaborn for visualization

Install the project's dependencies:

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_advanced_ai.txt
```

PyTorch installation can vary by operating system and CPU/GPU setup. If the command above cannot find a compatible PyTorch build, install PyTorch using the instructions at [pytorch.org](https://pytorch.org/get-started/locally/) and then install the remaining requirements.

## Quick start

Run the built-in demonstration:

```bash
python advanced_ai_memory_system.py
```

Or use the system from Python:

```python
from advanced_ai_memory_system import AIConfig, AdvancedAIMemorySystem

system = AdvancedAIMemorySystem(AIConfig())
system.start_system()

try:
    result = system.process_interaction(
        "How can memory help an AI system?",
        context={"session_id": "demo"},
    )
    print(result["response"])
    print(result["consciousness_metrics"])
finally:
    system.stop_system()
```

The reported metrics are diagnostic outputs of this prototype, not validated measures of consciousness.

## Project files

| File | Purpose |
| --- | --- |
| `advanced_ai_memory_system.py` | Core system, memory, knowledge, monitoring, and adaptation components |
| `advanced_ai_visualization.py` | Visualizations for the advanced system |
| `memory_visualization.py` | Memory-oriented visualization utilities |
| `test_advanced_ai_system.py` | Core system demonstrations and tests |
| `test_visualization.py` | Visualization tests |
| `requirements_advanced_ai.txt` | Python dependencies |
| `README_ADVANCED_AI.md` | Earlier, more detailed project notes |
| `HONEST_SYSTEM_COMPARISON.md` | Notes on project scope and limitations |

## Tests

Run the tests from the project directory:

```bash
python -m pytest -q
```

Visualization tests may create image files and runtime data in the working directory.

## Generated data and checkpoints

Runtime databases, logs, test outputs, visualizations, Python caches, and model checkpoints are generated artifacts and are excluded from this source repository by `.gitignore`. In particular, the local training checkpoints are multi-gigabyte files and are not included. Keep a separate backup of any generated data or checkpoints you need; this repository is intended for source code and project documentation.

## Scope and limitations

This is a research prototype, not a production-ready monitoring or safety system. Its behavior metrics are heuristic, optional model integrations depend on separately installed libraries and model assets, and the project should not be used to make claims about an AI system's inner experience.
