# Multi-PDF Research Assistant (Pro)

## What it does
A local, multi-PDF **RAG-style** research assistant:
- Indexes multiple PDFs in a directory
- Builds embeddings for chunked text
- Retrieves relevant chunks with vector search (FAISS)
- Answers your question
  - If `OPENAI_API_KEY` is set: uses OpenAI to synthesize an answer with citations
  - Otherwise: returns the top retrieved passages + citations

## Repo structure
- `research_assistant/research_assistant.py` — CLI + implementation
- `research_assistant/requirements.txt` — dependencies

## Install
```bash
pip install -r requirements.txt
```

## Usage
### 1) Create an index
```bash
python research_assistant.py --index --pdf_dir ./pdfs --index_dir ./index
```

### 2) Ask a question
```bash
python research_assistant.py --ask "What is backpropagation?" --index_dir ./index
```

### 3) (Optional) Inspect results
The assistant will print:
- Answer
- Citations: which PDF chunks were used

## Notes
- The first run downloads the sentence-transformer model.
- For best results, keep PDFs relatively clean (text-based PDFs).

