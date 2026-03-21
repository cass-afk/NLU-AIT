"""
A6: Contextual Retrieval Chatbot — Task 3
Chapter 2: Words and Tokens (Jurafsky & Martin, 2026)
Student: st125002

Models:
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Generator:  Qwen/Qwen2.5-3B-Instruct
- Vector Store: FAISS IndexFlatIP (cosine similarity)

Setup:
  Place contextual_chunks.json in the same folder as this file, then run:
  streamlit run app.py
"""

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR               = Path(__file__).parent
CONTEXTUAL_CHUNKS_PATH = BASE_DIR / "contextual_chunks.json"
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"
GEN_MODEL_NAME         = "Qwen/Qwen2.5-3B-Instruct"
TOP_K                  = 3

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chapter 2 QA — Words & Tokens",
    page_icon="📖",
    layout="wide"
)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📖 Chapter 2 QA")
    st.markdown("**Student:** st125002")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown(f"**Retriever:** `{EMBEDDING_MODEL}`")
    st.markdown(f"**Generator:** `{GEN_MODEL_NAME}`")
    st.markdown(f"**Method:** Contextual Retrieval")
    st.markdown("---")
    st.markdown(
        "Ask anything about **Chapter 2: Words and Tokens** "
        "from *Speech and Language Processing* (Jurafsky & Martin, 2026). "
        "\n\nEach answer cites the source chunk(s) used to generate it."
    )

# ─── Load chunks ───────────────────────────────────────────────────────────────
@st.cache_data
def load_chunks() -> List[Dict]:
    if not CONTEXTUAL_CHUNKS_PATH.exists():
        st.error(
            f"`contextual_chunks.json` not found in `{BASE_DIR}`.\n\n"
            "Copy it from `outputs/task2/contextual_chunks.json` into the `app/` folder."
        )
        st.stop()
    with open(CONTEXTUAL_CHUNKS_PATH, encoding="utf-8") as f:
        return json.load(f)

# ─── Embedding model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)

# ─── FAISS index ───────────────────────────────────────────────────────────────
@st.cache_resource
def build_faiss_index(_chunks: List[Dict], _embedder: SentenceTransformer) -> faiss.Index:
    """Build index on the 'text' field (context_prefix + original_text combined)."""
    texts = [c["text"] for c in _chunks]
    embeddings = _embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# ─── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(
    query: str,
    chunks: List[Dict],
    index: faiss.Index,
    embedder: SentenceTransformer,
    top_k: int = TOP_K
) -> List[Dict]:
    q_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    scores, indices = index.search(q_emb, top_k)
    return [
        {
            "rank":                rank,
            "score":               float(scores[0][i]),
            "chunk_id":            chunks[int(idx)]["chunk_id"],
            "source_paragraph_id": chunks[int(idx)]["source_paragraph_id"],
            "context_prefix":      chunks[int(idx)]["context_prefix"],
            "original_text":       chunks[int(idx)]["original_text"],
            "text":                chunks[int(idx)]["text"],
        }
        for i, (rank, idx) in enumerate(zip(range(1, top_k + 1), indices[0]))
    ]

# ─── Generator ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_generator():
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    mdl = AutoModelForCausalLM.from_pretrained(
        GEN_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    return tok, gen

def generate_answer(
    question: str,
    retrieved: List[Dict],
    tokenizer,
    gen_pipeline,
    max_new_tokens: int = 180,
    temperature: float = 0.2,
) -> str:
    context_text = "\n\n".join(
        f"[Chunk {r['chunk_id']}]\n{r['text']}" for r in retrieved
    )
    prompt = f"""You are answering a question using only the retrieved chapter context.

Rules:
1. Use ONLY the context below.
2. Do NOT use outside knowledge.
3. If the answer is not clearly supported by the context, say so briefly.
4. Write a concise answer in 1-3 sentences.

Question:
{question}

Retrieved Context:
{context_text}

Answer:"""

    messages = [
        {"role": "system", "content": "You are a careful academic assistant."},
        {"role": "user",   "content": prompt},
    ]
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = gen_pipeline(
        text_input,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=0.9,
        return_full_text=False,
    )
    return out[0]["generated_text"].strip()

# ─── Initialise ────────────────────────────────────────────────────────────────
with st.spinner("🔧 Loading models and building index (first run ~1–2 min)..."):
    chunks   = load_chunks()
    embedder = load_embedder()
    index    = build_faiss_index(chunks, embedder)
    tokenizer, gen_pipeline = load_generator()

st.success(f"Ready — {len(chunks)} contextual chunks indexed.")

# ─── Helper: render source chunk card ──────────────────────────────────────────
def render_source(src: Dict) -> None:
    st.markdown(
        f"**Chunk {src['chunk_id']}** &nbsp;|&nbsp; "
        f"Paragraph `{src['source_paragraph_id']}` &nbsp;|&nbsp; "
        f"Score: `{src['score']:.4f}`"
    )
    st.markdown(f"> 🏷️ *{src['context_prefix']}*")
    preview = src["original_text"]
    st.markdown(f"> {preview[:500]}{'...' if len(preview) > 500 else ''}")
    st.markdown("---")

# ─── Main UI ───────────────────────────────────────────────────────────────────
st.title("📖 Chapter 2: Words and Tokens — QA Chatbot")
st.markdown(
    "Ask about **tokenization, BPE, morphemes, Unicode, "
    "regular expressions, edit distance**, and more from the textbook."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Source chunks used"):
                for src in msg["sources"]:
                    render_source(src)

# New question
if question := st.chat_input("Ask something about Chapter 2..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and generating..."):
            results = retrieve(question, chunks, index, embedder)
            answer  = generate_answer(question, results, tokenizer, gen_pipeline)

        st.markdown(answer)
        with st.expander("📄 Source chunks used"):
            for src in results:
                render_source(src)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": results,
    })