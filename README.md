# RAG Evaluation Demo

A compact Retrieval-Augmented Generation demo that pairs a FAISS-backed
document retriever with RAGAS evaluation metrics. It is built around small
sample files in `data/` so the retrieval and scoring flow is easy to inspect.

This is a learning/demo project, not a production document platform.

## What it does

- Loads a toy knowledge base from `data/knowledge.txt`.
- Splits the knowledge base into paragraph chunks.
- Embeds chunks with OpenAI embeddings.
- Stores chunks in an in-memory FAISS vector store.
- Launches a Gradio chatbot for questions over the retrieved context.
- Runs a single-turn RAGAS evaluation from `data/evaluation.txt`.

## Tech Stack

- Python
- Gradio
- OpenAI chat and embedding models
- LangChain / LangChain OpenAI
- FAISS
- RAGAS

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your own OpenAI API key.

## Run

```bash
python3 main.py
```

Choose:

- `1` for the chatbot
- `2` for the RAGAS evaluation run

## RAGAS Metrics

The evaluation path currently scores:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

## Current Limitations

- Uses a small static toy dataset.
- Rebuilds the FAISS index at startup.
- Uses a command-line choice before launching chatbot or evaluation mode.
- Evaluation uses one sample from `data/evaluation.txt`, not a full benchmark suite.
- No authentication, persistence, uploaded files, or production UI.

## Public Safety Notes

Do not commit `.env`, `.vectorstore/`, `.ragas_cache/`, `.venv/`, or generated
cache files. The real API key should only live in your local `.env`.
