"""research_assistant.py

Multi-PDF Research Assistant (Pro)

Features
- Index multiple PDFs from a folder
- Extract text, chunk with overlap
- Build local embeddings (sentence-transformers)
- Store embeddings in a FAISS index (cached on disk)
- Retrieve top-k relevant chunks for a question
- Answer with citations
  - If OPENAI_API_KEY is set: synthesize an answer using OpenAI
  - Otherwise: return top retrieved passages as the answer (extractive)

CLI
- python research_assistant.py --index --pdf_dir ./pdfs --index_dir ./index
- python research_assistant.py --ask "..." --index_dir ./index
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChunkMeta:
    chunk_id: int
    pdf_path: str
    pdf_name: str
    page: Optional[int]
    start_char: int
    end_char: int


def extract_pdf_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """Return list of (page_number, page_text).

    NOTE: Some PDFs (scanned/image PDFs) have no extractable text layer.
    This assistant currently tries text-layer extraction only; callers should
    fall back to OCR if needed.


    This function tries multiple extraction backends because some PDFs expose
    different text layers.
    """

    # Backend 1: pypdf
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        PdfReader = None  # type: ignore

    if PdfReader is not None:
        reader = PdfReader(str(pdf_path))
        pages: List[Tuple[int, str]] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            text = re.sub(r"\s+", " ", text).strip()
            pages.append((i, text))

        # If we extracted any non-empty text, return immediately
        if any(len(t.strip()) > 0 for _, t in pages):
            return pages

    # Backend 2: pdfminer.six (fallback)
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PDF text extraction failed. Install dependencies: pypdf and pdfminer.six"
        ) from e

    # pdfminer extracts whole document; we cannot reliably map to pages,
    # so we return a single 'page' chunk.
    full_text = pdfminer_extract_text(str(pdf_path)) or ""
    full_text = re.sub(r"\s+", " ", full_text).strip()
    return [(0, full_text)] if full_text else [(0, "")]



def chunk_text(
    pages: List[Tuple[int, str]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Chunk page texts into overlapping windows.

    Primary strategy: character-window chunking with overlap.
    Fallback: if a PDF is sparse/oddly-formatted and yields 0 chunks,
    we switch to line-based chunking (still with citations metadata).
    """

    chunks: List[Tuple[str, Dict[str, Any]]] = []

    def _add_chunk(text: str, page_no: int, start_char: int, end_char: int) -> None:
        t = (text or "").strip()
        if not t:
            return
        chunks.append(
            (
                t,
                {
                    "page": page_no,
                    "start_char": int(start_char),
                    "end_char": int(end_char),
                },
            )
        )

    # 1) Character-window chunking
    for page_no, page_text in pages:
        page_text = page_text or ""
        if len(page_text.strip()) == 0:
            continue

        start = 0
        n = len(page_text)
        while start < n:
            end = min(n, start + chunk_size)
            chunk = page_text[start:end]
            _add_chunk(chunk, page_no, start, end)
            if end >= n:
                break
            # Prevent infinite loops if overlap >= chunk_size
            start = max(0, end - min(chunk_overlap, max(0, chunk_size - 1)))

    # If chunking produced nothing, force at least one chunk per non-empty page
    if not chunks:
        for page_no, page_text in pages:
            page_text = (page_text or "").strip()
            if not page_text:
                continue
            _add_chunk(page_text, page_no, 0, len(page_text))
            # Keep it bounded
            if len(chunks) >= 2000:
                break


    if chunks:
        return chunks

    # 2) Fallback: line-based chunking (robust to PDFs with poor char layout)
    for page_no, page_text in pages:
        page_text = page_text or ""
        if len(page_text.strip()) == 0:
            continue

        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        if not lines:
            continue

        buf: List[str] = []
        buf_len = 0
        start_char = 0
        # Approximate char ranges by cumulative length
        cumulative = 0
        for ln in lines:
            ln_len = len(ln) + 1
            if buf_len + ln_len > chunk_size and buf:
                chunk_str = "\n".join(buf)
                _add_chunk(chunk_str, page_no, start_char, start_char + len(chunk_str))
                # overlap via last N chars of previous chunk
                buf = [buf_str for buf_str in buf][-1:]
                buf = [buf[-1]]
                buf_len = len(buf[0])
                start_char = max(0, cumulative - chunk_overlap)
            buf.append(ln)
            buf_len += ln_len
            cumulative += ln_len

        if buf:
            chunk_str = "\n".join(buf)
            _add_chunk(chunk_str, page_no, start_char, start_char + len(chunk_str))

    return chunks



def _ensure_embeddings_matrix(all_embeddings: List[Any]) -> "Any":
    """Convert list of embeddings to a strict (num_chunks, dim) float32 matrix."""
    import numpy as np

    if not all_embeddings:
        raise RuntimeError("No embeddings were generated.")

    # Typical: each item is (dim,) or (1, dim). Normalize to (dim,)
    vecs: List[np.ndarray] = []
    for e in all_embeddings:
        arr = np.asarray(e, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 1:
            raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
        vecs.append(arr)

    dim = int(vecs[0].shape[0])
    for v in vecs:
        if int(v.shape[0]) != dim:
            raise RuntimeError("Embedding dimension mismatch across chunks.")

    return np.stack(vecs, axis=0)  # (num_chunks, dim)


def build_or_load_index(
    *,
    pdf_dir: Path,
    index_dir: Path,
    embedding_model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    force_reindex: bool,
) -> None:
    """Build embeddings + FAISS index + caches if not already present."""

    index_dir.mkdir(parents=True, exist_ok=True)

    faiss_path = index_dir / "faiss.index"
    meta_path = index_dir / "chunks_meta.json"
    state_path = index_dir / "index_state.json"
    chunk_text_path = index_dir / "chunks_text.pkl"

    if (
        not force_reindex
        and faiss_path.exists()
        and meta_path.exists()
        and state_path.exists()
        and chunk_text_path.exists()
    ):
        return

    pdf_paths = sorted([p for p in pdf_dir.glob("**/*.pdf") if p.is_file()])
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {pdf_dir}")

    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore
    import numpy as np

    model = SentenceTransformer(embedding_model_name)

    all_embeddings: List[Any] = []
    chunks_meta: List[ChunkMeta] = []
    chunks_text: List[str] = []

    chunk_id = 0
    for pdf_path in pdf_paths:
        pages = extract_pdf_text(pdf_path)
        chunks = chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_text_, m in chunks:
            emb = model.encode([chunk_text_], normalize_embeddings=True)
            # emb can be shape (1, dim) or sometimes (dim,)
            all_embeddings.append(np.asarray(emb))
            chunks_text.append(chunk_text_)
            chunks_meta.append(
                ChunkMeta(
                    chunk_id=chunk_id,
                    pdf_path=str(pdf_path),
                    pdf_name=pdf_path.name,
                    page=m.get("page"),
                    start_char=m.get("start_char"),
                    end_char=m.get("end_char"),
                )
            )
            chunk_id += 1

    vectors = _ensure_embeddings_matrix(all_embeddings)  # (num_chunks, dim)

    dim = int(vectors.shape[1])
    index = faiss.IndexFlatIP(dim)  # embeddings normalized => cosine via inner product
    index.add(vectors)

    faiss.write_index(index, str(faiss_path))

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump([c.__dict__ for c in chunks_meta], f, ensure_ascii=False)

    with chunk_text_path.open("wb") as f:
        pickle.dump(chunks_text, f)

    with state_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pdf_dir": str(pdf_dir),
                "num_pdfs": len(pdf_paths),
                "num_chunks": len(chunks_meta),
                "embedding_model": embedding_model_name,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "force_reindex": force_reindex,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_index(index_dir: Path) -> Tuple[Any, List[ChunkMeta], List[str]]:
    faiss_path = index_dir / "faiss.index"
    meta_path = index_dir / "chunks_meta.json"
    chunk_text_path = index_dir / "chunks_text.pkl"

    if not (faiss_path.exists() and meta_path.exists() and chunk_text_path.exists()):
        raise FileNotFoundError(f"Index not found in {index_dir}. Run with --index first.")

    import faiss  # type: ignore

    index = faiss.read_index(str(faiss_path))

    with meta_path.open("r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    chunks_meta = [ChunkMeta(**m) for m in meta_raw]

    with chunk_text_path.open("rb") as f:
        chunks_text = pickle.load(f)

    if len(chunks_text) != len(chunks_meta):
        # Keep robust: still allow, but should align.
        pass

    return index, chunks_meta, chunks_text


def retrieve(
    *,
    question: str,
    index_dir: Path,
    embedding_model_name: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    """Return retrieved chunks with their text."""

    import numpy as np
    from sentence_transformers import SentenceTransformer  # type: ignore

    faiss_index, chunks_meta, chunks_text = load_index(index_dir)

    model = SentenceTransformer(embedding_model_name)
    q_emb = model.encode([question], normalize_embeddings=True)
    q_vec = np.asarray(q_emb, dtype=np.float32)
    if q_vec.ndim == 2 and q_vec.shape[0] == 1:
        q_vec = q_vec[0]
    q_vec = q_vec.astype("float32")

    scores, indices = faiss_index.search(q_vec.reshape(1, -1), top_k)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(chunks_meta):
            continue
        c = chunks_meta[idx]
        results.append(
            {
                "chunk_id": c.chunk_id,
                "pdf_path": c.pdf_path,
                "pdf_name": c.pdf_name,
                "page": c.page,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "score": float(score),
                "text": chunks_text[c.chunk_id] if c.chunk_id < len(chunks_text) else "",
            }
        )

    return results


def answer_with_openai(question: str, contexts: List[str], citations: List[str]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Install it in requirements.") from e

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are a precise research assistant. Use the provided contexts to answer the question. "
        "If the answer is not contained in the contexts, say you don't know. "
        "Provide short, factual answers with citations in the form [1], [2], ..."
    )

    context_blocks = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])
    user_prompt = (
        f"Question: {question}\n\nContexts:\n{context_blocks}\n\n"
        "Citations list (for reference):\n" + "\n".join([f"[{i+1}] {c}" for i, c in enumerate(citations)])
    )

    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    return resp.choices[0].message.content.strip()  # type: ignore


def fallback_extractive_answer(question: str, results: List[Dict[str, Any]], top_n: int) -> str:
    top = results[:top_n]
    parts = [f"Top relevant passages for: {question}\n"]
    for i, r in enumerate(top, start=1):
        parts.append(f"[{i}] {r['text']}")
    return "\n\n".join(parts).strip()


def query_pdfs(
    *,
    question: str,
    index_dir: Path,
    embedding_model_name: str,
    top_k: int,
    answer_top_n: int,
) -> None:
    results = retrieve(
        question=question,
        index_dir=index_dir,
        embedding_model_name=embedding_model_name,
        top_k=top_k,
    )

    contexts = [r["text"] for r in results]
    citations = [
        f"{r['pdf_name']} (page {r['page']}) chars {r['start_char']}–{r['end_char']}" for r in results
    ]

    try:
        answer = answer_with_openai(question, contexts=contexts, citations=citations)
        mode = "openai"
    except Exception:
        answer = fallback_extractive_answer(question, results, top_n=answer_top_n)
        mode = "extractive"

    print("\n=== Answer ===\n")
    print(answer)

    print("\n=== Citations ===\n")
    for i, c in enumerate(citations[:answer_top_n], start=1):
        print(f"[{i}] {c}")

    print(f"\n(mode: {mode})")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-PDF Research Assistant (Pro)")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", action="store_true", help="Index PDFs from pdf_dir")
    mode.add_argument("--ask", type=str, default=None, help="Ask a question")

    p.add_argument("--pdf_dir", type=str, default="./pdfs", help="Folder containing PDFs")
    p.add_argument("--index_dir", type=str, default="./index", help="Folder to store/load index")

    p.add_argument("--top_k", type=int, default=6, help="Top-k chunks to retrieve")
    p.add_argument("--answer_top_n", type=int, default=3, help="How many passages to show/cite")

    p.add_argument("--chunk_size", type=int, default=800, help="Chunk size in characters")
    p.add_argument("--chunk_overlap", type=int, default=120, help="Chunk overlap in characters")

    p.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model name",
    )

    p.add_argument("--force_reindex", action="store_true", help="Rebuild index even if cached")

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    pdf_dir = Path(args.pdf_dir)
    index_dir = Path(args.index_dir)

    if args.index:
        if not pdf_dir.exists():
            raise FileNotFoundError(f"--pdf_dir does not exist: {pdf_dir}")

        print(f"Indexing PDFs in: {pdf_dir}")
        print(f"Saving index to: {index_dir}")

        build_or_load_index(
            pdf_dir=pdf_dir,
            index_dir=index_dir,
            embedding_model_name=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force_reindex=args.force_reindex,
        )

        print("Indexing complete.")
        return

    if args.ask is not None:
        query_pdfs(
            question=args.ask,
            index_dir=index_dir,
            embedding_model_name=args.embedding_model,
            top_k=args.top_k,
            answer_top_n=args.answer_top_n,
        )
        return

    raise RuntimeError("No mode selected")


if __name__ == "__main__":
    main()

