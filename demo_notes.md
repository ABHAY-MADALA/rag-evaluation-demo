# Demo Notes

Recommended GitHub screenshots:

1. Gradio chatbot answering a question about ERR-9942 or Aether-V9.
2. Terminal output from option `2` showing the four RAGAS metric scores.
3. The sample `data/evaluation.txt` prompt/context/target format.

Suggested demo script:

1. Create `.env` from `.env.example` and add an OpenAI API key.
2. Run `python3 main.py`.
3. Choose `1` and ask: "How do I clear the Paradox Leak error code ERR-9942?"
4. Restart, choose `2`, and capture the RAGAS metric output.
5. Explain that the project demonstrates the mechanics of retrieval plus evaluation on toy data, not a production RAG benchmark.
