# GenAI w/ LanChain Examples and Utilities

This repository provides a set of examples and utilities demonstrating how to use Generative AI (GenAI) with LangChain. The examples showcase diverse functionalities, such as handling structured outputs, building conversational apps, and performing retrieval-augmented generation (RAG).

Feel free to fork, clone, and reuse these examples as you see fit. Each script is well-documented and ready to adapt to your needs.

## Prerequisites

- Python 3.11 or higher
- pip package manager

## Setting Up the Environment

It's encouraged to use a virtual environment to manage dependencies.

### Steps to Set Up:

1. Create and activate a virtual environment:

    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    source .venv/bin/activate  # On Windows PowerShell: .venv\Scripts\activate
                               # On Windows cmd.exe: .venv\Scripts\activate.bat
    ```

2. Install the project with all dependencies (including dev tools):

    ```bash
    pip install -e .[dev]
    ```

## Overview of Scripts

### Streamlit Apps

1. **`01_example_pdf_qa.py`**: Adds documents to a vector store and enables querying via a chatbot interface.
2. **`02_example_pdf_and_url_qa.py`**: Extends the previous app by allowing knowledge base updates from web sources.
3. **`04_example_chat_memory_and_grades.py`**: Provides a chatbot interface with graded retrieval and response evaluation.

#### Usage for Streamlit Apps:

```plaintext
  usage: streamlit run <script_name>.py -- [-h] [--inference-provider {nvidia,openai,ollama,anthropic}]
  -h, --help            show this help message and exit
  --inference-provider {nvidia,openai,ollama,anthropic}
                        The inference provider to use.
```

### Non-Streamlit Scripts

1. **`03_example_structured_output.py`**: Processes survey data and generates structured JSON output.
2. **`05_example_rag_agent_graph.py`**: Implements a Retrieval-Augmented Generation (RAG) agent that routes questions to vector store or web search, retrieves relevant documents, and generates responses.
3. **`06_example_structured_vision_understanding.py`**: Processes images to extract structured data, such as object descriptions, counts, and main objects.

**NOTE:** To implement the web-search tool, `05_example_rag_agent_graph.py` requires a `TAVILY_API_KEY` environment variable (which is an API key to https://tavily.com/).

### General Usage for Non-Streamlit Scripts:

```plaintext
  usage: <script_name>.py [-h] [--inference-provider {nvidia,openai,ollama,anthropic}]
  -h, --help            show this help message and exit
  --inference-provider {nvidia,openai,ollama,anthropic}
                        The inference provider to use.
```
#### (Exception) Usage for `06_example_structured_vision_understanding.py`:

```plaintext
  usage: 06_example_structured_vision_understanding.py [-h] [--inference-provider {nvidia,openai,ollama,anthropic}]
  positional arguments:
    input                 Path to the image file

  options:
    -h, --help            show this help message and exit
    --debug               Enable debug mode
    --inference-provider {nvidia,openai,ollama,anthropic}
                          The inference provider to use.
```

### Supported Inference Providers

The scripts support the following inference providers:

- `nvidia`
- `openai`
- `anthropic`
- `ollama`

## Provider-Specific Keys and Defaults

### NVIDIA

#### NVIDIA External (w/ rate limits)

- **Default LLM Model:** `meta/llama-3.3-70b-instruct`
- **Default Embedding Model:** `NV-Embed-QA`
- **Default Vision Model:** `meta/llama-3.2-90b-vision-instruct`
- **Environment Variable:** `NVIDIA_API_KEY` (required - get [HERE](https://build.nvidia.com/))

#### Steps to Obtain NVIDIA API Key

1. **Log in to [NVIDIA NGC](https://ngc.nvidia.com/) with your NVIDIA credentials.**
2. **Select your NVIDIA organization (choose 'NV-Developer').**
3. **Click on your account -> Setup -> Generate Personal Key -> +Generate Personal Key.**
4. **Name the key.**
5. **Set the expiration date.**
6. **Set Service Included to 'Cloud Functions'.**
7. **Click on 'Generate Personal Key'.**
8. **Save the key securely.**

### OpenAI

- **Default LLM Model:** `gpt-4o-mini`
- **Default Embedding Model:** `text-embedding-3-small`
- **Default Vision Model:** `gpt-4o-mini`
- **Environment Variable:** `OPENAI_API_KEY` (required - get [HERE](https://platform.openai.com/settings/organization/api-keys))

### Anthropic

- **Default LLM Model:** `claude-3-5-sonnet-20241022`
- **Default Embedding Model:** `voyage-3-lite`
- **Default Vision Model:** `claude-3-5-sonnet-20241022`
- **Environment Variables:**
  - `ANTHROPIC_API_KEY` (required - get [HERE](https://console.anthropic.com/settings/keys))
  - `VOYAGE_API_KEY` (required for embeddings - get [HERE](https://dashboard.voyageai.com/api-keys))

**NOTE:** Anthropic do not supply embedding models, so the `voyage` embedding model is used instead.

### Ollama (Local Server)

- **Default LLM Model:** `llama3.1:8b`
- **Default Embedding Model:** `mxbai-embed-large:latest`
- **Default Vision Model:** `llava-phi3:3.8b`
- **Note:** Requires Ollama server running.
