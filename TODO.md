# TODO - Multi-PDF Research Assistant (Pro Upgrade)

- [ ] Update `research_assistant/requirements.txt` with PDF parsing + embeddings + vector search deps
- [x] Replace `research_assistant/research_assistant.py` with full implementation:
  - [x] CLI: `--index` and `--ask`
  - [x] PDF ingestion + text extraction
  - [x] Chunking + overlap
  - [x] Local embeddings + caching (embeddings + metadata saved to disk)
  - [x] Vector retrieval (FAISS)
  - [x] Answering:
    - [x] OpenAI synthesis if `OPENAI_API_KEY` exists
    - [x] Otherwise return top passages + citations
- [x] Update `research_assistant/README.md` with usage instructions
- [x] Run a quick sanity check by installing deps + running `--help`



