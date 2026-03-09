# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["HF_DATASETS_OFFLINE"]  = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]              = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]   = "./hf_cache"

# ================= IMPORTS =================
import re
import sys
import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from typing import List, Tuple, Optional

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH  = r"models\qwen"
ADAPTER_PATH     = str(BASE_DIR / "models" / "qwen_qlora_adapters")
EMBED_PATH       = r"models\BAAI-bge-large"

# ── Cross-encoder path ─────────────────────────────────────────────────────
# Download once:  sentence-transformers download cross-encoder/ms-marco-MiniLM-L-6-v2
# Then set this path to wherever you saved it locally.
CROSS_ENCODER_PATH = r"models\ms-marco-MiniLM-L-6-v2"

CHROMA_DIR      = str(BASE_DIR / "chroma_store")
COLLECTION_NAME = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ================= LOAD BI-ENCODER (embedder) =================
print("Loading bi-encoder embedder (OFFLINE)...")
embedder = SentenceTransformer(EMBED_PATH, device="cuda", local_files_only=True)
print("✅ Bi-encoder loaded")

# ================= LOAD CROSS-ENCODER (reranker) =================
print("Loading cross-encoder reranker (OFFLINE)...")
cross_encoder_available = False
reranker = None

try:
    reranker = CrossEncoder(
        CROSS_ENCODER_PATH,
        max_length=512,
        device=device,
    )
    cross_encoder_available = True
    print("✅ Cross-encoder reranker loaded")
except Exception as e:
    print(f"⚠  Cross-encoder not loaded: {e}")
    print("   Falling back to bi-encoder + keyword boost reranking.")
    print(f"   To fix: download 'cross-encoder/ms-marco-MiniLM-L-6-v2' and set CROSS_ENCODER_PATH")

# ================= LOAD CHROMA =================
print("Loading ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Collection loaded — {collection.count()} chunks")

space = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"   Distance metric: {space}")
if space != "cosine":
    print(
        f"\n⛔  FATAL: Collection metric is '{space}', expected 'cosine'.\n"
        "    Re-run embeddings.py to recreate the collection with the correct metric.\n"
    )
    sys.exit(1)

# ================= LOAD FINE-TUNED LLM =================
print("Loading Qwen model...")
use_finetuned = os.path.exists(ADAPTER_PATH)

if use_finetuned:
    print("✓ Adapters found — loading with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ Fine-tuned Qwen loaded")
else:
    print("⚠  No adapters — loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    print("✅ Base Qwen loaded")

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, local_files_only=True, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


# =============================================================================
#  RETRIEVAL STAGE 1 — Bi-encoder candidate retrieval
# =============================================================================

def biencoder_retrieve(
    query: str,
    k: int = 20,
    verbose: bool = True,
) -> Tuple[List[str], List[float], List[dict]]:
    """
    Stage 1: Use the bi-encoder + ChromaDB to get top-k candidates quickly.
    Returns (docs, cosine_similarities, metadatas).
    """
    q_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    if verbose:
        print(f"  Bi-encoder query vector norm: {np.linalg.norm(q_vec[0]):.4f}")

    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[{}] * len(docs)])[0]

    # Convert cosine distance → cosine similarity
    similarities = [1.0 - (d / 2.0) for d in distances]

    if verbose:
        print(f"  Retrieved {len(docs)} candidates from ChromaDB")

    return docs, similarities, metadatas


# =============================================================================
#  RETRIEVAL STAGE 2 — Cross-encoder reranking
# =============================================================================

def crossencoder_rerank(
    query: str,
    docs: List[str],
    bi_scores: List[float],
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Stage 2: Rerank candidates using cross-encoder relevance scores.

    Cross-encoder reads (query, document) together — much more accurate than
    bi-encoder cosine similarity because it models interaction between query
    and document tokens directly.

    Returns (reranked_docs, reranker_scores) sorted best-first.
    """
    if not docs:
        return [], []

    # Build (query, doc) pairs for cross-encoder
    pairs = [(query, doc) for doc in docs]

    # Score all pairs — cross-encoder returns raw logits (higher = more relevant)
    ce_scores = reranker.predict(pairs, show_progress_bar=False)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {'Rank':<5} {'CE Score':<12} {'Bi-Sim':<10} {'Document preview':<40}")
        print(f"{'─'*70}")

    # Combine bi-encoder + cross-encoder scores
    # CE score is dominant; bi-encoder acts as a tie-breaker
    combined = []
    for i, (doc, ce_score, bi_score) in enumerate(zip(docs, ce_scores, bi_scores)):
        combined.append({
            "doc":      doc,
            "ce_score": float(ce_score),
            "bi_score": float(bi_score),
            # Normalise bi_score contribution so CE dominates
            "final":    float(ce_score) * 0.85 + float(bi_score) * 0.15,
        })

    # Sort by final score descending
    combined.sort(key=lambda x: x["final"], reverse=True)

    if verbose:
        for rank, item in enumerate(combined[:top_k], 1):
            preview = item["doc"][:42] + "..." if len(item["doc"]) > 42 else item["doc"]
            print(f"  {rank:<5} {item['ce_score']:<12.4f} {item['bi_score']:<10.4f} {preview}")
        print(f"{'─'*70}")

    top_docs   = [item["doc"]      for item in combined[:top_k]]
    top_scores = [item["ce_score"] for item in combined[:top_k]]

    return top_docs, top_scores


# =============================================================================
#  RETRIEVAL STAGE 2 (fallback) — keyword-boosted bi-encoder reranking
# =============================================================================

def biencoder_rerank(
    query: str,
    docs: List[str],
    bi_scores: List[float],
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Fallback reranker when cross-encoder is not available.
    Boosts bi-encoder scores with keyword overlap.
    """
    query_terms = set(query.lower().split())
    candidates  = []

    for doc, score in zip(docs, bi_scores):
        doc_lower = doc.lower()
        matches   = sum(1 for t in query_terms if len(t) > 3 and t in doc_lower)
        boost     = min(0.08 * matches, 0.2)
        candidates.append({
            "doc":    doc,
            "score":  min(score + boost, 1.0),
            "raw":    score,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {'Rank':<5} {'Boosted':<10} {'Raw Sim':<10} {'Document preview':<40}")
        print(f"{'─'*70}")
        for i, c in enumerate(candidates[:top_k], 1):
            preview = c["doc"][:42] + "..." if len(c["doc"]) > 42 else c["doc"]
            print(f"  {i:<5} {c['score']:<10.4f} {c['raw']:<10.4f} {preview}")
        print(f"{'─'*70}")

    filtered = [c for c in candidates[:top_k] if c["score"] >= threshold]
    return [c["doc"] for c in filtered], [c["score"] for c in filtered]


# =============================================================================
#  FULL RETRIEVAL PIPELINE  (bi-encoder → reranker)
# =============================================================================

def retrieve_and_rerank(
    query: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Full two-stage retrieval:
      Stage 1 — bi-encoder fetches `candidate_k` chunks from ChromaDB (fast)
      Stage 2 — cross-encoder reranks and returns best `top_k` (accurate)

    If cross-encoder is unavailable, falls back to keyword-boosted bi-encoder.

    Args:
        query       : user question
        candidate_k : how many candidates to pull from ChromaDB (cast wide)
        top_k       : how many to return after reranking
        threshold   : minimum score to accept a chunk
        verbose     : print debug tables
    """
    print(f"\n📡 Stage 1 — Bi-encoder retrieval (k={candidate_k})...")
    docs, bi_scores, _ = biencoder_retrieve(query, k=candidate_k, verbose=verbose)

    if not docs:
        return [], []

    # Hard check: if best bi-encoder score is terrible, bail early
    if bi_scores and bi_scores[0] < 0.35:
        print(f"⚠  Very low bi-encoder score ({bi_scores[0]:.3f}) — query may be out of domain")
        return [], []

    if cross_encoder_available:
        print(f"🎯 Stage 2 — Cross-encoder reranking (top_k={top_k})...")
        reranked_docs, reranked_scores = crossencoder_rerank(
            query, docs, bi_scores, top_k=top_k, verbose=verbose
        )

        # Cross-encoder scores are logits — no fixed threshold; take top_k as-is
        # but drop anything the CE scored very low (< -5 is clearly irrelevant)
        filtered_docs   = []
        filtered_scores = []
        for doc, score in zip(reranked_docs, reranked_scores):
            if score > -5.0:
                filtered_docs.append(doc)
                filtered_scores.append(score)

        print(f"✅ Cross-encoder selected {len(filtered_docs)} chunks")
        return filtered_docs, filtered_scores

    else:
        print(f"🔄 Stage 2 — Keyword-boosted reranking (fallback, top_k={top_k})...")
        return biencoder_rerank(
            query, docs, bi_scores,
            top_k=top_k, threshold=threshold, verbose=verbose
        )


# =============================================================================
#  CONTEXT CLEANING
# =============================================================================

def clean_context(context: str, preserve_structure: bool = True) -> str:
    cleaned    = []
    lines      = context.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            if preserve_structure and cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        lower = line.lower()
        if any(p in lower for p in [
            "page intentionally left blank",
            "this page intentionally",
            "continued on next page",
        ]):
            continue

        if re.match(r'^(\d+\.|\*|-|•|[A-Z][A-Z\s]+:)', line):
            cleaned.append(line)
            continue

        if any(c in line for c in ['=', ':', '(', ')', '%', '@']):
            cleaned.append(line)
            continue

        if len(line) < 20 and not any(c.isdigit() for c in line):
            continue

        cleaned.append(line)

    final      = []
    prev_empty = False
    for line in cleaned:
        if line == "":
            if not prev_empty:
                final.append(line)
            prev_empty = True
        else:
            final.append(line)
            prev_empty = False

    return "\n".join(final)


# =============================================================================
#  QUESTION TYPE DETECTION
# =============================================================================

def detect_question_type(question: str) -> str:
    q = question.lower()

    if any(p in q for p in [
        "in points", "as points", "point by point", "in bullet", "as bullets",
        "bullet points", "numbered list", "as a list", "list format",
        "step by step", "in steps", "one by one", "enumerate",
    ]):
        return "points"

    if any(q.startswith(p) for p in ["how do", "how to", "how does", "procedure", "steps", "process"]):
        return "procedure"

    if any(q.startswith(p) for p in ["what is", "what are", "define", "meaning of"]):
        return "definition"

    if any(q.startswith(p) for p in ["explain", "describe", "tell me about", "overview", "elaborate"]):
        return "explanation"

    if any(w in q for w in ["list", "mention all", "what are all", "give all", "types of", "kinds of"]):
        return "list"

    if any(w in q for w in ["safety", "measures", "precautions", "guidelines", "requirements", "rules"]):
        return "list"

    if any(w in q for w in ["why", "reason", "cause", "purpose", "benefit", "advantage", "disadvantage"]):
        return "explanation"

    if any(w in q for w in ["compare", "difference", "vs", "versus", "distinguish", "contrast"]):
        return "comparison"

    return "general"


# =============================================================================
#  PROMPT BUILDER
# =============================================================================

def build_prompt(
    context: str,
    question: str,
    qtype: str,
    similarity_score: Optional[float] = None,
) -> str:
    format_instructions = {
        "definition": (
            "Write a clear, precise definition. "
            "Use exact terms from the context. Keep it concise (2-4 sentences)."
        ),
        "explanation": (
            "Write a detailed explanation in clear paragraphs. "
            "Cover all key aspects mentioned in the context."
        ),
        "procedure": (
            "List the steps as a numbered list:\n"
            "1. First step\n2. Second step\n...\n"
            "Use only steps described in the context."
        ),
        "points": (
            "Answer using a numbered list. Each point must be a complete, "
            "informative sentence drawn directly from the context.\n"
            "Format:\n1. [Point one]\n2. [Point two]\n3. [Point three]\n...\n"
            "Do NOT write paragraphs — use ONLY the list format."
        ),
        "list": (
            "List all relevant items mentioned in the context. "
            "Use a numbered or bulleted format with a brief explanation for each."
        ),
        "comparison": (
            "Compare the items by highlighting their key differences and similarities "
            "as described in the context. Use structured points if helpful."
        ),
        "general": (
            "Answer comprehensively using specific information from the context. "
            "Be direct and complete."
        ),
    }

    style_instruction = format_instructions.get(qtype, format_instructions["general"])

    confidence_note = ""
    if similarity_score is not None and similarity_score < 0.55:
        confidence_note = (
            "\nNote: Context relevance is moderate. "
            "Answer only what is explicitly supported by the context."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant. Answer questions using "
                "ONLY the information in the provided context.\n\n"
                "RULES:\n"
                "1. Use ONLY facts explicitly stated in the context.\n"
                "2. Do NOT add outside knowledge or assumptions.\n"
                "3. If context is insufficient, say: 'The available context does not "
                "contain sufficient information to answer this.'\n"
                "4. Mirror the vocabulary and terminology used in the context.\n"
                "5. Follow the response format instruction exactly.\n\n"
                f"RESPONSE FORMAT: {style_instruction}"
                f"{confidence_note}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER:"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
#  ANSWER CLEANING & VALIDATION
# =============================================================================

def clean_generated_answer(answer: str) -> str:
    for marker in ["<|im_start|>assistant", "<|im_start|>", "<|im_end|>"]:
        if marker in answer:
            parts  = answer.split(marker)
            answer = parts[-1] if marker == "<|im_start|>assistant" else parts[0]

    answer = re.sub(r'^(assistant|system|user)\s*[:\n]', '', answer, flags=re.IGNORECASE).strip()

    lines         = answer.splitlines()
    cleaned_lines = []
    skip_echo     = True
    for line in lines:
        stripped = line.strip()
        if skip_echo and (not stripped or stripped.endswith("?")):
            continue
        skip_echo = False
        cleaned_lines.append(line)

    answer = "\n".join(cleaned_lines).strip()
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    return answer.strip()


def validate_answer(answer: str) -> bool:
    if not answer or len(answer.split()) < 5:
        return False

    refusal_phrases = [
        "as an ai language model",
        "i don't have access to",
        "based on my training data",
        "i cannot provide information",
        "my knowledge cutoff",
    ]
    if any(p in answer.lower() for p in refusal_phrases):
        return False

    if len(answer.strip()) < 20:
        return False

    return True


def ensure_complete_sentence(text: str) -> str:
    if not text:
        return text
    text = text.strip()
    if text and text[-1] not in ".!?":
        last_line = text.splitlines()[-1].strip() if text.splitlines() else ""
        if re.match(r'^\d+\.', last_line) or last_line.startswith(('-', '•', '*')):
            return text
        if len(text.split()) > 5:
            return text + "."
    return text


def validate_context_relevance(context: str, question: str) -> float:
    question_terms = set(question.lower().split())
    context_lower  = context.lower()
    meaningful     = [t for t in question_terms if len(t) > 3]
    if not meaningful:
        return 1.0
    matches = sum(1 for t in meaningful if t in context_lower)
    return matches / len(meaningful)


# =============================================================================
#  ANSWER GENERATION
# =============================================================================

def generate_answer(
    context: str,
    question: str,
    similarity_score: Optional[float] = None,
) -> str:
    qtype   = detect_question_type(question)
    context = clean_context(context, preserve_structure=True)

    if not context.strip() or len(context.split()) < 15:
        return "Insufficient context available to answer this question."

    relevance = validate_context_relevance(context, question)
    if relevance < 0.05:
        return "The retrieved context does not appear to contain relevant information for this question."

    prompt = build_prompt(context, question, qtype, similarity_score)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.4,
            top_p=0.90,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    answer = clean_generated_answer(answer)

    if not validate_answer(answer):
        return "Unable to provide a reliable answer based on the available context."

    return answer


# =============================================================================
#  MAIN RAG PIPELINE
# =============================================================================

def ask(
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
) -> str:
    """
    Full RAG pipeline:
      1. Bi-encoder retrieves `candidate_k` candidates from ChromaDB
      2. Cross-encoder reranks → selects best `top_k`  (or keyword-boost fallback)
      3. Deduplicate chunks
      4. Generate answer with format-aware prompting
    """
    print(f"\n{'='*60}")
    print(f"QUERY : {question}")
    print(f"TYPE  : {detect_question_type(question)}")
    print(f"RERANK: {'Cross-encoder' if cross_encoder_available else 'Keyword-boost (fallback)'}")
    print('='*60)

    docs, scores = retrieve_and_rerank(
        question,
        candidate_k=candidate_k,
        top_k=top_k,
        threshold=threshold,
        verbose=True,
    )

    # Retry with looser candidate pool if nothing found
    if not docs:
        print(f"⚠  No results — retrying with larger candidate pool (k={candidate_k + 10})...")
        docs, scores = retrieve_and_rerank(
            question,
            candidate_k=candidate_k + 10,
            top_k=top_k + 3,
            threshold=max(threshold - 0.10, 0.25),
            verbose=False,
        )

    if not docs:
        return "I don't have sufficient information in the knowledge base to answer this question."

    # Deduplicate
    seen        = set()
    unique_docs = []
    for doc in docs:
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n---\n\n".join(unique_docs)

    if len(context.split()) < 20:
        return "The retrieved information is too limited to provide a reliable answer."

    # Use cross-encoder score for confidence hint if available;
    # otherwise fall back to bi-encoder similarity
    avg_score = sum(scores) / len(scores) if scores else 0
    answer    = generate_answer(context, question, avg_score)

    return ensure_complete_sentence(answer)


# =============================================================================
#  STARTUP DIAGNOSTIC
# =============================================================================

def run_startup_diagnostic():
    print("\n" + "="*60)
    print(" STARTUP DIAGNOSTIC")
    print("="*60)

    try:
        sample = collection.get(limit=10, include=["embeddings", "documents"])

        if not sample["embeddings"]:
            print("❌ ERROR: No embeddings found!")
            return

        embed_dim = len(sample["embeddings"][0])
        print(f"✓ Embedding dimension : {embed_dim}")

        norms    = [np.linalg.norm(e) for e in sample["embeddings"]]
        avg_norm = np.mean(norms)
        print(f"✓ Vector norms        : mean={avg_norm:.4f}, std={np.std(norms):.4f}")

        if not all(0.95 < n < 1.05 for n in norms[:5]):
            print("⚠ Vectors may not be properly normalized!")
        else:
            print("✓ Vectors are properly normalized")

        # Self-retrieval test
        print("\n📊 Self-Retrieval Test:")
        test_doc     = sample["documents"][0]
        test_snippet = test_doc[:200]

        q_vec = embedder.encode(
            [test_snippet], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        res = collection.query(
            query_embeddings=q_vec, n_results=3,
            include=["documents", "distances"]
        )
        if res["documents"][0]:
            dist = res["distances"][0][0]
            sim  = 1.0 - (dist / 2.0)
            print(f"  Distance: {dist:.4f}  |  Similarity: {sim:.4f}")
            if res["documents"][0][0][:100] == test_doc[:100]:
                print("✓ Exact match — retrieval working correctly")
            elif sim > 0.9:
                print("✓ High similarity match — retrieval working")
            else:
                print("⚠ Low self-retrieval similarity — check embedding model")

        # Cross-encoder sanity test
        if cross_encoder_available:
            print("\n🎯 Cross-Encoder Test:")
            test_pairs = [
                ("What is the system configuration?", test_doc[:300]),
                ("What is the system configuration?", "This is completely unrelated text about apples."),
            ]
            ce_scores = reranker.predict(test_pairs, show_progress_bar=False)
            print(f"  Relevant doc score   : {ce_scores[0]:.4f}")
            print(f"  Irrelevant doc score : {ce_scores[1]:.4f}")
            if ce_scores[0] > ce_scores[1]:
                print("✓ Cross-encoder correctly ranks relevant doc higher")
            else:
                print("⚠ Cross-encoder ranking unexpected — check model")
        else:
            print("\n⚠  Cross-encoder not available — using fallback reranker")

        print("\n" + "="*60)
        print(" DIAGNOSTIC COMPLETE")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    run_startup_diagnostic()

    model_type = "Fine-tuned" if use_finetuned else "Base"
    rerank_type = "Cross-encoder" if cross_encoder_available else "Keyword-boost (fallback)"

    print(f"\n✅ RAG System Ready")
    print(f"   Model      : {model_type} Qwen2-1.5B")
    print(f"   Device     : {device.upper()}")
    print(f"   Chunks     : {collection.count()}")
    print(f"   Reranker   : {rerank_type}")
    print("\nType 'exit' or 'quit' to stop.")
    print("Type 'help' for usage tips.\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not q:
            continue

        if q.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if q.lower() == "help":
            print("\nUsage Tips:")
            print("  - Ask specific questions about the content in your PDF/docs")
            print("  - You can ask 'explain in points', 'list all', 'step by step'")
            print("  - Use keywords from your training material")
            print("  - Rephrased questions work — e.g. 'what does X do?' vs 'explain X'\n")
            continue

        answer = ask(q, candidate_k=20, top_k=5, threshold=0.45)

        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")