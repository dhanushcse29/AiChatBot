# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]         = "1"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]                = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "./hf_cache"

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

BASE_MODEL_PATH    = r"models\llama3.2-3b"
ADAPTER_PATH       = str(BASE_DIR / "models" / "llama3.2_qlora_adapters")
EMBED_PATH         = r"models\BAAI-bge-large"
CROSS_ENCODER_PATH = r"models\ms-marco-MiniLM-L-6-v2"
CHROMA_DIR         = str(BASE_DIR / "chroma_store")
COLLECTION_NAME    = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ================= LOAD BI-ENCODER =================
print("Loading bi-encoder embedder (OFFLINE)...")
embedder = SentenceTransformer(EMBED_PATH, device=device, local_files_only=True)
print("✅ Bi-encoder loaded")

# ================= LOAD CROSS-ENCODER =================
print("Loading cross-encoder reranker (OFFLINE)...")
cross_encoder_available = False
reranker = None

try:
    reranker = CrossEncoder(CROSS_ENCODER_PATH, max_length=512, device=device)
    cross_encoder_available = True
    print("✅ Cross-encoder reranker loaded")
except Exception as e:
    print(f"⚠  Cross-encoder not loaded: {e}")
    print("   Falling back to bi-encoder + keyword boost reranking.")

# ================= LOAD CHROMA =================
print("Loading ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Collection loaded — {collection.count()} chunks")

space = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"   Distance metric: {space}")
if space != "cosine":
    print(f"\n⛔  FATAL: Collection metric is '{space}', expected 'cosine'.")
    sys.exit(1)

# ================= LOAD LLAMA 3.2 3B =================
print("\nLoading Llama 3.2 3B model...")
use_finetuned = os.path.exists(ADAPTER_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if use_finetuned:
    print("✓ Adapters found — loading fine-tuned Llama 3.2 3B with QLoRA...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ Fine-tuned Llama 3.2 3B loaded")
else:
    print("⚠  No adapters found — loading base Llama 3.2 3B...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    print("✅ Base Llama 3.2 3B loaded")

model.eval()

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, local_files_only=True
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
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
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
    if not docs:
        return [], []

    pairs     = [(query, doc) for doc in docs]
    ce_scores = reranker.predict(pairs, show_progress_bar=False)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {'Rank':<5} {'CE Score':<12} {'Bi-Sim':<10} {'Document preview':<40}")
        print(f"{'─'*70}")

    combined = []
    for doc, ce_score, bi_score in zip(docs, ce_scores, bi_scores):
        combined.append({
            "doc":      doc,
            "ce_score": float(ce_score),
            "bi_score": float(bi_score),
            "final":    float(ce_score) * 0.85 + float(bi_score) * 0.15,
        })

    combined.sort(key=lambda x: x["final"], reverse=True)

    if verbose:
        for rank, item in enumerate(combined[:top_k], 1):
            preview = item["doc"][:42] + "..." if len(item["doc"]) > 42 else item["doc"]
            print(f"  {rank:<5} {item['ce_score']:<12.4f} {item['bi_score']:<10.4f} {preview}")
        print(f"{'─'*70}")

    return [item["doc"] for item in combined[:top_k]], [item["ce_score"] for item in combined[:top_k]]


# =============================================================================
#  RETRIEVAL STAGE 2 (fallback) — Keyword-boosted bi-encoder reranking
# =============================================================================

def biencoder_rerank(
    query: str,
    docs: List[str],
    bi_scores: List[float],
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    query_terms = set(query.lower().split())
    candidates  = []

    for doc, score in zip(docs, bi_scores):
        doc_lower = doc.lower()
        matches   = sum(1 for t in query_terms if len(t) > 3 and t in doc_lower)
        boost     = min(0.08 * matches, 0.2)
        candidates.append({"doc": doc, "score": min(score + boost, 1.0), "raw": score})

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
#  FULL RETRIEVAL PIPELINE
# =============================================================================

def retrieve_and_rerank(
    query: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    print(f"\n📡 Stage 1 — Bi-encoder retrieval (k={candidate_k})...")
    docs, bi_scores, _ = biencoder_retrieve(query, k=candidate_k, verbose=verbose)

    if not docs:
        return [], []

    if bi_scores and bi_scores[0] < 0.35:
        print(f"⚠  Very low bi-encoder score ({bi_scores[0]:.3f}) — query may be out of domain")
        return [], []

    if cross_encoder_available:
        print(f"🎯 Stage 2 — Cross-encoder reranking (top_k={top_k})...")
        reranked_docs, reranked_scores = crossencoder_rerank(
            query, docs, bi_scores, top_k=top_k, verbose=verbose
        )
        filtered_docs   = [d for d, s in zip(reranked_docs, reranked_scores) if s > -5.0]
        filtered_scores = [s for s in reranked_scores if s > -5.0]
        print(f"✅ Cross-encoder selected {len(filtered_docs)} chunks")
        return filtered_docs, filtered_scores
    else:
        print(f"🔄 Stage 2 — Keyword-boosted reranking (fallback, top_k={top_k})...")
        return biencoder_rerank(
            query, docs, bi_scores, top_k=top_k, threshold=threshold, verbose=verbose
        )


# =============================================================================
#  CONTEXT CLEANING
# =============================================================================

def clean_context(context: str, preserve_structure: bool = True) -> str:
    cleaned = []
    for line in context.splitlines():
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
            cleaned.append(line); continue
        if any(c in line for c in ['=', ':', '(', ')', '%', '@']):
            cleaned.append(line); continue
        if len(line) < 20 and not any(c.isdigit() for c in line):
            continue
        cleaned.append(line)

    final, prev_empty = [], False
    for line in cleaned:
        if line == "":
            if not prev_empty: final.append(line)
            prev_empty = True
        else:
            final.append(line); prev_empty = False
    return "\n".join(final)


# =============================================================================
#  QUESTION TYPE DETECTION
# =============================================================================

def detect_question_type(question: str) -> str:
    q = question.lower()
    if any(p in q for p in ["in points", "as points", "bullet points", "numbered list", "step by step"]):
        return "points"
    if any(q.startswith(p) for p in ["how do", "how to", "how does", "procedure", "steps"]):
        return "procedure"
    if any(q.startswith(p) for p in ["what is", "what are", "define", "meaning of"]):
        return "definition"
    if any(q.startswith(p) for p in ["explain", "describe", "tell me about", "overview"]):
        return "explanation"
    if any(w in q for w in ["list", "mention all", "types of", "kinds of", "safety", "precautions"]):
        return "list"
    if any(w in q for w in ["why", "reason", "cause", "purpose", "benefit"]):
        return "explanation"
    if any(w in q for w in ["compare", "difference", "vs", "versus", "contrast"]):
        return "comparison"
    return "general"


# =============================================================================
#  PROMPT BUILDER — Llama 3.2 chat template via tokenizer
# =============================================================================

def build_prompt(
    context: str,
    question: str,
    qtype: str,
    similarity_score: Optional[float] = None,
) -> str:
    format_instructions = {
        "definition":  "Write a clear, precise definition in 2-4 sentences using exact terms from the context.",
        "explanation": "Write a detailed explanation in clear paragraphs covering all key aspects in the context.",
        "procedure":   "List the steps as a numbered list (1. First step, 2. Second step ...) using only steps from the context.",
        "points":      "Answer using a numbered list. Each point must be a complete sentence from the context. Do NOT write paragraphs.",
        "list":        "List all relevant items from the context using a numbered or bulleted format with brief explanations.",
        "comparison":  "Compare the items by highlighting key differences and similarities from the context.",
        "general":     "Answer comprehensively using specific information from the context. Be direct and complete.",
    }

    style = format_instructions.get(qtype, format_instructions["general"])
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
                f"RESPONSE FORMAT: {style}"
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

    # tokenizer.apply_chat_template handles Llama 3.2 special tokens automatically
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
#  ANSWER CLEANING & VALIDATION
# =============================================================================

def clean_generated_answer(answer: str) -> str:
    for token in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>"]:
        answer = answer.replace(token, "")

    answer = re.sub(r'^(assistant|system|user)\s*[:\n]', '', answer, flags=re.IGNORECASE).strip()

    lines, skip_echo = [], True
    for line in answer.splitlines():
        stripped = line.strip()
        if skip_echo and (not stripped or stripped.endswith("?")):
            continue
        skip_echo = False
        lines.append(line)

    answer = "\n".join(lines).strip()
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    return answer.strip()


def validate_answer(answer: str) -> bool:
    if not answer or len(answer.split()) < 5:
        return False
    refusal_phrases = [
        "as an ai language model", "i don't have access to",
        "based on my training data", "i cannot provide information", "my knowledge cutoff",
    ]
    if any(p in answer.lower() for p in refusal_phrases):
        return False
    return len(answer.strip()) >= 20


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
    return sum(1 for t in meaningful if t in context_lower) / len(meaningful)


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

    if validate_context_relevance(context, question) < 0.05:
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
    print(f"\n{'='*60}")
    print(f"QUERY : {question}")
    print(f"TYPE  : {detect_question_type(question)}")
    print(f"RERANK: {'Cross-encoder' if cross_encoder_available else 'Keyword-boost (fallback)'}")
    print('='*60)

    docs, scores = retrieve_and_rerank(
        question, candidate_k=candidate_k, top_k=top_k, threshold=threshold, verbose=True
    )

    if not docs:
        print(f"⚠  No results — retrying with larger candidate pool...")
        docs, scores = retrieve_and_rerank(
            question,
            candidate_k=candidate_k + 10,
            top_k=top_k + 3,
            threshold=max(threshold - 0.10, 0.25),
            verbose=False,
        )

    if not docs:
        return "I don't have sufficient information in the knowledge base to answer this question."

    seen, unique_docs = set(), []
    for doc in docs:
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n---\n\n".join(unique_docs)

    if len(context.split()) < 20:
        return "The retrieved information is too limited to provide a reliable answer."

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

        norms = [np.linalg.norm(e) for e in sample["embeddings"]]
        print(f"✓ Embedding dimension : {len(sample['embeddings'][0])}")
        print(f"✓ Vector norms        : mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")
        print("✓ Vectors normalized" if all(0.95 < n < 1.05 for n in norms[:5]) else "⚠ Vectors may not be normalized!")

        print("\n📊 Self-Retrieval Test:")
        test_doc = sample["documents"][0]
        q_vec = embedder.encode([test_doc[:200]], normalize_embeddings=True, convert_to_numpy=True).tolist()
        res   = collection.query(query_embeddings=q_vec, n_results=3, include=["documents", "distances"])
        if res["documents"][0]:
            sim = 1.0 - (res["distances"][0][0] / 2.0)
            print(f"  Similarity: {sim:.4f}")
            print("✓ Retrieval working correctly" if sim > 0.9 else "⚠ Low self-retrieval score")

        if cross_encoder_available:
            print("\n🎯 Cross-Encoder Test:")
            scores = reranker.predict([
                ("What is the system configuration?", test_doc[:300]),
                ("What is the system configuration?", "Apples grow on trees in orchards."),
            ], show_progress_bar=False)
            print(f"  Relevant score   : {scores[0]:.4f}")
            print(f"  Irrelevant score : {scores[1]:.4f}")
            print("✓ Cross-encoder ranking correct" if scores[0] > scores[1] else "⚠ Unexpected ranking")

        print("\n🤖 Generation Test (short)...")
        test_ans = generate_answer("The sky is blue due to Rayleigh scattering.", "Why is the sky blue?")
        print(f"  Response: {test_ans[:120]}...")
        print("✓ Generation working")

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

    model_type  = "Fine-tuned" if use_finetuned else "Base"
    rerank_type = "Cross-encoder" if cross_encoder_available else "Keyword-boost (fallback)"

    print(f"\n✅ RAG System Ready")
    print(f"   Model      : {model_type} Llama 3.2 3B")
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
            print("  - Use 'explain in points', 'list all', 'step by step'")
            print("  - Rephrased questions work fine\n")
            continue

        answer = ask(q, candidate_k=20, top_k=5, threshold=0.45)
        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")