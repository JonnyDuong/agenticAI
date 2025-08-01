# Agentic AI Repository

This repository contains various experiments and projects related to agentic AI systems.

## EXP1: Mistral White Paper RAG System

The first experiment in this repository is a Retrieval-Augmented Generation (RAG) system that provides information about the Mistral AI white paper.

### Overview

EXP1 uses:
- **FAISS** for efficient vector similarity search
- **OpenRouter** to access powerful LLMs like Mixtral 8x7b
- **LangChain** for agent framework and tooling
- **Flask** for creating a web API
- **HuggingFace Embeddings** for semantic encoding

### Key Features

- Semantic search over the Mistral AI white paper
- Web interface for querying the system
- API endpoints for integration with other applications
- Chunked document processing for better retrieval
- Context-aware responses using LangChain agents

### How It Works

1. The system breaks down the Mistral paper into semantic chunks
2. User queries are encoded into vector embeddings
3. FAISS retrieves the most relevant text chunks
4. An LLM (via OpenRouter) generates a comprehensive answer using the retrieved context
5. The answer is returned via web interface or API

### Getting Started

See the [EXP1 README](./EXP1/README.md) for detailed setup and usage instructions.

## Future Experiments

More experiments will be added to this repository over time, exploring different aspects of agentic AI systems.

## License

This project is open-sourced under the MIT license.
