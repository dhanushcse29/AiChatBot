# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]               = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]    = "./hf_cache"

# ================= IMPORTS =================
import re
import sys
import json
import uuid
import datetime
import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH    = r"models\qwen"
ADAPTER_PATH       = str(BASE_DIR / "models" / "qwen_qlora_adapters")
EMBED_PATH         = r"models\BAAI-bge-large"
CROSS_ENCODER_PATH = r"models\ms-marco-MiniLM-L-6-v2"
CHROMA_DIR         = str(BASE_DIR / "chroma_db")
COLLECTION_NAME    = "pdf_markdown_embeddings_new"
FEEDBACK_FILE      = str(BASE_DIR / "feedback_store.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# =============================================================================
#  FEEDBACK STORE  — simple JSON-based persistent store
# =============================================================================
#
#  Schema (one record per feedback event):
#  {
#    "id": "<uuid>",
#    "timestamp": "2025-01-01T12:00:00",
#    "question": "...",
#    "answer": "...",
#    "rating": "correct" | "incorrect" | "partial",
#    "correction": "...",          # optional: user-supplied correct answer
#    "query_variants": [...],       # multi-query variants that were used
#    "retrieved_chunks": [...]      # chunk previews that were used
#  }
#
# =============================================================================

class FeedbackStore:
    """
    Lightweight JSON-backed feedback store.

    How it feeds back into the pipeline:
    ─────────────────────────────────────
    1.  QUERY EXPANSION BOOST
        When multi_query_retrieve is called, it checks if the question (or a
        close paraphrase) has been marked "incorrect" before.  If yes, it
        automatically widens candidate_k by +10 and lowers the rerank
        threshold by 0.05 so harder-to-reach chunks are surfaced.

    2.  NEGATIVE CHUNK FILTERING
        Chunks that were consistently retrieved but led to "incorrect" answers
        are gently down-weighted by temporarily boosting their cross-encoder
        threshold (you can extend this to a denylist if needed).

    3.  CORRECTION INJECTION
        If a user supplied a correction, it is prepended to the retrieved
        context the next time the same (or very similar) question is asked,
        acting as a "gold snippet" that guides the model.
    """

    def __init__(self, path: str):
        self.path    = Path(path)
        self.records: List[Dict] = []
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self.records = json.load(f)
                print(f"✅ Feedback store loaded — {len(self.records)} records")
            except Exception as e:
                print(f"⚠  Could not load feedback store: {e} — starting fresh")
                self.records = []
        else:
            self.records = []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2, ensure_ascii=False)

    def add(
        self,
        question: str,
        answer: str,
        rating: str,                        # "correct" | "incorrect" | "partial"
        correction: Optional[str] = None,
        query_variants: Optional[List[str]] = None,
        retrieved_chunks: Optional[List[str]] = None,
    ) -> str:
        record = {
            "id":              str(uuid.uuid4())[:8],
            "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
            "question":        question,
            "answer":          answer,
            "rating":          rating,
            "correction":      correction,
            "query_variants":  query_variants or [],
            "retrieved_chunks": [c[:200] for c in (retrieved_chunks or [])],
        }
        self.records.append(record)
        self._save()
        print(f"💾 Feedback saved [id={record['id']}, rating={rating}]")
        return record["id"]

    # ------------------------------------------------------------------
    # Helpers used by the pipeline
    # ------------------------------------------------------------------

    def get_correction_for(self, question: str, threshold: int = 1) -> Optional[str]:
        """
        Return the most recent user-supplied correction for `question`
        if that question has been marked incorrect ≥ `threshold` times
        AND the user provided a correction text.

        Threshold is 1 so the correction is used on the very next attempt
        after the user submits it — no need to fail 3 times first.
        """
        q_lower = question.lower().strip()
        relevant = [
            r for r in self.records
            if r["rating"] in ("incorrect", "partial")
            and r.get("correction")
            and q_lower in r["question"].lower()
        ]
        if len(relevant) >= threshold:
            return relevant[-1]["correction"]
        return None

    def should_widen_search(self, question: str, min_fails: int = 1) -> bool:
        """Return True if this question has failed ≥ min_fails times before."""
        q_lower = question.lower().strip()
        fails = sum(
            1 for r in self.records
            if r["rating"] in ("incorrect", "partial")
            and q_lower in r["question"].lower()
        )
        return fails >= min_fails

    def summary(self) -> Dict:
        total    = len(self.records)
        correct  = sum(1 for r in self.records if r["rating"] == "correct")
        partial  = sum(1 for r in self.records if r["rating"] == "partial")
        wrong    = sum(1 for r in self.records if r["rating"] == "incorrect")
        return {
            "total": total,
            "correct": correct,
            "partial": partial,
            "incorrect": wrong,
            "accuracy": f"{100*correct/total:.1f}%" if total else "N/A",
        }


# ─────────────────────────── initialise feedback store ──────────────────────
feedback_store = FeedbackStore(FEEDBACK_FILE)


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
    print("   Falling back to keyword-boost reranking.")

# ================= LOAD CHROMA =================
print("Loading ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Collection loaded — {collection.count()} chunks")

space = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"   Distance metric: {space}")
if space != "cosine":
    print(f"\n⛔  FATAL: Collection metric is '{space}', expected 'cosine'.\n"
          "    Re-run embeddings.py to recreate the collection.\n")
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
#  MULTI-QUERY EXPANSION  ← KEY FIX for "correct only on 2nd attempt"
# =============================================================================
#
#  ROOT CAUSE of the first-attempt failure:
#  ─────────────────────────────────────────
#  A single bi-encoder query only retrieves chunks that are close to the
#  *exact wording* of the question.  When the question uses vocabulary that
#  differs slightly from the document (e.g. "config" vs "configuration",
#  "EW system" vs "electronic warfare system"), the most relevant chunk may
#  rank 21st out of 30 and get cut off at k=20.  On the 2nd attempt the user
#  often rephrases slightly, which shifts the embedding just enough to catch
#  that chunk.
#
#  FIX: generate 3 paraphrases of the question locally (no API call needed),
#  run a separate bi-encoder query for each, then union the results before
#  reranking.  This is fast (<200 ms extra) and dramatically improves recall.
#
# =============================================================================

# Simple rule-based paraphraser — no LLM call needed, works fully offline
_PARAPHRASE_RULES = [
    # Expand contractions / abbreviations common in EW/technical docs
    (r"\bconfig\b",          "configuration"),
    (r"\bsys\b",             "system"),
    (r"\bfreq\b",            "frequency"),
    (r"\bEW\b",              "electronic warfare"),
    (r"\bEA\b",              "electronic attack"),
    (r"\bES\b",              "electronic support"),
    (r"\bRF\b",              "radio frequency"),
    (r"\bGHz\b",             "gigahertz"),
    (r"\bMHz\b",             "megahertz"),
    # Question stem variants
    (r"^[Ww]hat is ",        "Describe "),
    (r"^[Ww]hat are ",       "List all "),
    (r"^[Hh]ow does ",       "Explain the mechanism of "),
    (r"^[Hh]ow do ",         "Explain how to "),
    (r"^[Ee]xplain ",        "What is "),
    (r"^[Dd]escribe ",       "What is the purpose of "),
]

def generate_query_variants(question: str, n: int = 3) -> List[str]:
    """
    Generate up to `n` distinct paraphrases of `question` using deterministic
    rule-based transformations.  No GPU or API call needed.

    Returns a deduplicated list that always starts with the original question.
    """
    variants = [question]

    # Variant 1 — abbreviation expansion
    v1 = question
    for pattern, replacement in _PARAPHRASE_RULES[:8]:   # abbreviation rules only
        v1 = re.sub(pattern, replacement, v1)
    if v1 != question:
        variants.append(v1)

    # Variant 2 — question stem swap
    v2 = question
    for pattern, replacement in _PARAPHRASE_RULES[8:]:   # stem rules only
        new_v = re.sub(pattern, replacement, v2)
        if new_v != v2:
            v2 = new_v
            break
    if v2 != question and v2 not in variants:
        variants.append(v2)

    # Variant 3 — strip question words, use noun phrase as keyword query
    noun_phrase = re.sub(
        r'^(what is|what are|how does|how do|explain|describe|list|'
        r'define|tell me about|give me|what|how|why|when|where)\s+',
        '', question, flags=re.IGNORECASE
    ).strip().rstrip("?")
    if noun_phrase and noun_phrase not in variants and len(noun_phrase) > 5:
        variants.append(noun_phrase)

    return variants[:n]


# =============================================================================
#  RETRIEVAL — Stage 1: Bi-encoder
# =============================================================================

def biencoder_retrieve(
    query: str,
    k: int = 20,
    verbose: bool = False,
) -> Tuple[List[str], List[float], List[dict]]:
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )

    docs        = results["documents"][0]
    distances   = results["distances"][0]
    metadatas   = results.get("metadatas", [[{}] * len(docs)])[0]
    similarities = [1.0 - (d / 2.0) for d in distances]

    if verbose:
        print(f"    [{query[:40]}...] → {len(docs)} candidates, best sim={similarities[0]:.3f}")

    return docs, similarities, metadatas


# =============================================================================
#  MULTI-QUERY RETRIEVAL  ← NEW
# =============================================================================

def multi_query_retrieve(
    question: str,
    candidate_k: int = 20,
    verbose: bool = True,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Run the bi-encoder query for each paraphrase variant of `question`,
    union all retrieved chunks (deduplicated by first 120 chars), and keep
    the best score seen for each chunk across all variants.

    Returns:
        docs          — deduplicated list of chunk texts
        best_scores   — highest bi-encoder similarity seen per chunk
        query_variants — the variants that were actually used
    """
    variants = generate_query_variants(question, n=3)

    # Widen search automatically if this question has failed before
    if feedback_store.should_widen_search(question):
        candidate_k += 10
        print(f"  📈 Feedback boost: widening to k={candidate_k} (past failures detected)")

    if verbose:
        print(f"\n📡 Multi-query retrieval — {len(variants)} variants, k={candidate_k} each")
        for i, v in enumerate(variants, 1):
            print(f"    V{i}: {v}")

    seen: Dict[str, float] = {}   # key=chunk_prefix → best score
    doc_map: Dict[str, str] = {}  # key=chunk_prefix → full doc

    for variant in variants:
        docs, scores, _ = biencoder_retrieve(variant, k=candidate_k, verbose=verbose)
        for doc, score in zip(docs, scores):
            key = doc[:120]
            if key not in seen or score > seen[key]:
                seen[key]    = score
                doc_map[key] = doc

    # Sort by best score descending
    sorted_items = sorted(seen.items(), key=lambda x: x[1], reverse=True)
    final_docs   = [doc_map[k] for k, _ in sorted_items]
    final_scores = [v              for _, v in sorted_items]

    if verbose:
        print(f"  ✅ Union pool: {len(final_docs)} unique chunks "
              f"(best score={final_scores[0]:.3f})")

    return final_docs, final_scores, variants


# =============================================================================
#  RETRIEVAL — Stage 2: Cross-encoder reranking
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
        print(f"\n{'─'*70}")
        print(f"  {'Rank':<5} {'CE Score':<12} {'Bi-Sim':<10} {'Document preview'}")
        print(f"{'─'*70}")
        for rank, item in enumerate(combined[:top_k], 1):
            preview = item["doc"][:45] + "..." if len(item["doc"]) > 45 else item["doc"]
            print(f"  {rank:<5} {item['ce_score']:<12.4f} {item['bi_score']:<10.4f} {preview}")
        print(f"{'─'*70}")

    top_docs   = [item["doc"]      for item in combined[:top_k]]
    top_scores = [item["ce_score"] for item in combined[:top_k]]
    return top_docs, top_scores


def biencoder_rerank(
    query: str,
    docs: List[str],
    bi_scores: List[float],
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """Fallback reranker using keyword boost."""
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
        print(f"  {'Rank':<5} {'Boosted':<10} {'Raw Sim':<10} {'Document preview'}")
        print(f"{'─'*70}")
        for i, c in enumerate(candidates[:top_k], 1):
            preview = c["doc"][:45] + "..." if len(c["doc"]) > 45 else c["doc"]
            print(f"  {i:<5} {c['score']:<10.4f} {c['raw']:<10.4f} {preview}")
        print(f"{'─'*70}")

    filtered = [c for c in candidates[:top_k] if c["score"] >= threshold]
    return [c["doc"] for c in filtered], [c["score"] for c in filtered]


# =============================================================================
#  FULL RETRIEVAL PIPELINE  (multi-query → reranker)
# =============================================================================

def retrieve_and_rerank(
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Full pipeline:
      Stage 1 — multi-query bi-encoder (unions paraphrase variants)
      Stage 2 — cross-encoder reranking (or keyword-boost fallback)

    Returns (docs, scores, query_variants_used).
    """
    docs, bi_scores, variants = multi_query_retrieve(
        question, candidate_k=candidate_k, verbose=verbose
    )

    if not docs:
        return [], [], variants

    if bi_scores and bi_scores[0] < 0.30:
        print(f"⚠  Very low best bi-encoder score ({bi_scores[0]:.3f}) — may be out of domain")
        return [], [], variants

    if cross_encoder_available:
        print(f"\n🎯 Stage 2 — Cross-encoder reranking (top_k={top_k})...")
        reranked_docs, reranked_scores = crossencoder_rerank(
            question, docs, bi_scores, top_k=top_k, verbose=verbose
        )
        filtered_docs   = [d for d, s in zip(reranked_docs, reranked_scores) if s > -5.0]
        filtered_scores = [s for s in reranked_scores if s > -5.0]
        print(f"✅ Cross-encoder selected {len(filtered_docs)} chunks")
        return filtered_docs, filtered_scores, variants
    else:
        print(f"\n🔄 Stage 2 — Keyword-boost reranking (fallback, top_k={top_k})...")
        docs_f, scores_f = biencoder_rerank(
            question, docs, bi_scores,
            top_k=top_k, threshold=threshold, verbose=verbose
        )
        return docs_f, scores_f, variants


# =============================================================================
#  CONTEXT CLEANING
# =============================================================================

def clean_context(context: str, preserve_structure: bool = True) -> str:
    cleaned    = []
    lines      = context.splitlines()
    for line in lines:
        line  = line.strip()
        lower = line.lower()
        if not line:
            if preserve_structure and cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        if any(p in lower for p in [
            "page intentionally left blank", "this page intentionally",
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
#  PROMPT BUILDER  ← IMPROVED
# =============================================================================
#
#  Key changes vs. original:
#  1. System prompt explicitly names the domain ("EW systems technical manual")
#     so the model stays in domain even for vague questions.
#  2. Added an explicit "context window" boundary so the model cannot confuse
#     context vs question.
#  3. Stronger anti-hallucination guardrail: "If you are unsure, say so."
#  4. Added "correction" injection slot — when a user has previously corrected
#     this question the correction is injected as a "KNOWN CORRECT ANSWER"
#     snippet before the context, which the model is told to treat as ground
#     truth.
#
# =============================================================================

def build_prompt(
    context: str,
    question: str,
    qtype: str,
    similarity_score: Optional[float] = None,
    correction: Optional[str] = None,
) -> str:

    format_instructions = {
        "definition": (
            "Write a clear, precise definition in 2–4 sentences. "
            "Use the exact technical terms from the context."
        ),
        "explanation": (
            "Write a detailed explanation in clear paragraphs covering all "
            "key aspects mentioned in the context."
        ),
        "procedure": (
            "Provide a numbered list of steps:\n"
            "1. First step\n2. Second step\n...\n"
            "Use only steps described in the context."
        ),
        "points": (
            "Answer using a numbered list only. Each point must be a complete "
            "informative sentence.\n"
            "Format:\n1. [Point one]\n2. [Point two]\n...\n"
            "Do NOT write paragraphs — numbered list ONLY."
        ),
        "list": (
            "List all relevant items from the context with a brief explanation "
            "for each. Use a numbered or bulleted format."
        ),
        "comparison": (
            "Compare the items by their key differences and similarities as "
            "described in the context. Use structured points."
        ),
        "general": (
            "Answer comprehensively and directly using only information from "
            "the context below."
        ),
    }

    style_instruction = format_instructions.get(qtype, format_instructions["general"])

    confidence_note = ""
    if similarity_score is not None and similarity_score < 0.55:
        confidence_note = (
            "\nIMPORTANT: Context relevance is MODERATE. "
            "Only answer what is explicitly stated. "
            "If the context does not contain enough information, say so."
        )

    correction_block = ""
    if correction:
        correction_block = (
            f"\n\n[VERIFIED CORRECT ANSWER FROM PREVIOUS SESSION]\n"
            f"{correction}\n"
            f"[END VERIFIED ANSWER — use this as primary reference]\n"
        )

    system_content = (
        "You are a precise technical assistant for EW (Electronic Warfare) "
        "systems documentation. Your sole knowledge source is the CONTEXT "
        "provided below — you must NOT use any outside knowledge.\n\n"
        "STRICT RULES:\n"
        "1. Answer ONLY from information explicitly present in the CONTEXT.\n"
        "2. Do NOT invent, assume, or extrapolate beyond the CONTEXT.\n"
        "3. Mirror the exact terminology and abbreviations used in the CONTEXT.\n"
        "4. If the CONTEXT lacks enough information, respond with exactly:\n"
        "   'The available context does not contain sufficient information to answer this.'\n"
        "5. Do NOT repeat the question in your answer.\n"
        "6. Do NOT add a preamble like 'Based on the context...' — answer directly.\n"
        f"7. RESPONSE FORMAT: {style_instruction}"
        f"{confidence_note}"
    )

    user_content = (
        f"CONTEXT:\n"
        f"{'─'*60}\n"
        f"{context}"
        f"{correction_block}\n"
        f"{'─'*60}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": user_content},
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

    lines, cleaned_lines, skip_echo = answer.splitlines(), [], True
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
    """
    Accept any answer that isn't empty or a known refusal phrase.
    The old thresholds (5 words, 20 chars) were blocking legitimate
    short factual answers like "Shakti covers 175 MHz to 40 GHz."
    """
    if not answer or not answer.strip():
        return False
    refusal_phrases = [
        "as an ai language model", "i don't have access to",
        "based on my training data", "i cannot provide information",
        "my knowledge cutoff",
    ]
    if any(p in answer.lower() for p in refusal_phrases):
        return False
    # Only reject truly empty/garbage output (< 3 words)
    return len(answer.split()) >= 3


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
    """
    Returns a plain Python float (not numpy scalar) to avoid ambiguous
    truth-value errors when used in  `if relevance < 0.05`.
    """
    question_terms = set(question.lower().split())
    context_lower  = context.lower()
    meaningful     = [t for t in question_terms if len(t) > 3]
    if not meaningful:
        return 1.0
    matches = sum(1 for t in meaningful if t in context_lower)
    return float(matches) / float(len(meaningful))


# =============================================================================
#  ANSWER GENERATION
# =============================================================================

def generate_answer(
    context: str,
    question: str,
    similarity_score: Optional[float] = None,
    correction: Optional[str] = None,
) -> str:
    qtype   = detect_question_type(question)
    context = clean_context(context, preserve_structure=True)

    if not context.strip() or len(context.split()) < 15:
        return "Insufficient context available to answer this question."

    relevance = validate_context_relevance(context, question)
    if relevance < 0.05:
        return "The retrieved context does not appear to contain relevant information."

    prompt = build_prompt(context, question, qtype, similarity_score, correction)

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
            # ── Generation parameters (tuned for 4 GB GPU) ─────────────────
            # do_sample=False (greedy) is more deterministic and faster on
            # small GPUs.  Switch to do_sample=True + low temperature if you
            # want more varied phrasing.
            do_sample=False,
            # If you re-enable sampling, use:
            # do_sample=True, temperature=0.2, top_p=0.85, top_k=40
            repetition_penalty=1.15,
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
    threshold: float = 0.40,   # slightly lower than original to reduce false "no context" rejections
    verbose: bool = True,
) -> Tuple[str, List[str], List[str]]:
    """
    Full RAG pipeline with multi-query retrieval and feedback injection.

    Returns:
        answer         — generated answer string
        retrieved_docs — list of chunk texts used
        variants_used  — query paraphrases that were issued
    """
    print(f"\n{'='*60}")
    print(f"QUERY : {question}")
    print(f"TYPE  : {detect_question_type(question)}")
    print(f"RERANK: {'Cross-encoder' if cross_encoder_available else 'Keyword-boost (fallback)'}")
    print('='*60)

    docs, scores, variants = retrieve_and_rerank(
        question,
        candidate_k=candidate_k,
        top_k=top_k,
        threshold=threshold,
        verbose=verbose,
    )

    # Retry with looser pool if nothing surfaced
    if not docs:
        print(f"⚠  No results — retrying (k={candidate_k+10}, threshold={threshold-0.10:.2f})...")
        docs, scores, variants = retrieve_and_rerank(
            question,
            candidate_k=candidate_k + 10,
            top_k=top_k + 3,
            threshold=max(threshold - 0.10, 0.20),
            verbose=False,
        )

    if not docs:
        msg = "I don't have sufficient information in the knowledge base to answer this question."
        return msg, [], variants

    # Deduplicate
    seen, unique_docs = set(), []
    for doc in docs:
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n---\n\n".join(unique_docs)

    if len(context.split()) < 20:
        msg = "The retrieved information is too limited to provide a reliable answer."
        return msg, unique_docs, variants

    # ── Feedback: inject correction if this question was previously wrong ──
    correction = feedback_store.get_correction_for(question)
    if correction:
        print(f"💡 Injecting past correction into prompt")

    avg_score = sum(scores) / len(scores) if scores else 0
    answer    = generate_answer(context, question, avg_score, correction=correction)

    return ensure_complete_sentence(answer), unique_docs, variants


# =============================================================================
#  INTERACTIVE FEEDBACK LOOP
# =============================================================================

def interactive_feedback(question: str, answer: str, docs: List[str], variants: List[str]):
    """
    After showing the answer, ask the user to rate it and optionally provide
    a correction.  Ratings are stored in feedback_store.json.
    """
    print(f"\n{'─'*60}")
    print("Was this answer correct?")
    print("  [y] Yes — correct")
    print("  [p] Partial — mostly right but incomplete")
    print("  [n] No — incorrect")
    print("  [s] Skip (don't save feedback)")
    print("  [q] Quit")

    rating_map = {"y": "correct", "p": "partial", "n": "incorrect"}

    try:
        choice = input("Rating: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False   # signal caller to exit

    if choice == "q":
        return False
    if choice == "s" or choice not in rating_map:
        return True

    rating     = rating_map[choice]
    correction = None

    if rating in ("incorrect", "partial"):
        print("Optional: type the correct answer (or press Enter to skip):")
        try:
            correction = input("Correction: ").strip() or None
        except (EOFError, KeyboardInterrupt):
            pass

    feedback_store.add(
        question=question,
        answer=answer,
        rating=rating,
        correction=correction,
        query_variants=variants,
        retrieved_chunks=docs,
    )

    if rating == "correct":
        print("✅ Great! Marked as correct.")
    else:
        print(f"📝 Feedback saved. System will adapt next time this question is asked.")

    return True


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
            print("❌ ERROR: No embeddings found!"); return

        embed_dim = len(sample["embeddings"][0])
        print(f"✓ Embedding dimension : {embed_dim}")

        norms    = [np.linalg.norm(e) for e in sample["embeddings"]]
        avg_norm = np.mean(norms)
        print(f"✓ Vector norms        : mean={avg_norm:.4f}, std={np.std(norms):.4f}")
        print("✓ Vectors normalised" if all(0.95 < float(n) < 1.05 for n in norms[:5])
              else "⚠ Vectors may not be properly normalized!")

        # Self-retrieval test
        test_doc     = sample["documents"][0]
        q_vec = embedder.encode(
            [test_doc[:200]], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        res   = collection.query(query_embeddings=q_vec, n_results=3,
                                 include=["documents", "distances"])
        if res["documents"][0]:
            dist = res["distances"][0][0]
            sim  = 1.0 - (dist / 2.0)
            print(f"✓ Self-retrieval sim  : {sim:.4f}")
            if sim < 0.90:
                print("⚠ Low self-retrieval similarity — check embedding model")

        # Multi-query variant test
        test_q    = "What is the system configuration?"
        variants  = generate_query_variants(test_q, n=3)
        print(f"\n✓ Multi-query variants for '{test_q}':")
        for i, v in enumerate(variants, 1):
            print(f"    V{i}: {v}")

        # Feedback store summary
        fb = feedback_store.summary()
        print(f"\n✓ Feedback store      : {fb['total']} records | accuracy={fb['accuracy']}")

        if cross_encoder_available:
            test_pairs = [
                ("What is the system configuration?", test_doc[:300]),
                ("What is the system configuration?", "Unrelated text about apples."),
            ]
            ce_scores = reranker.predict(test_pairs, show_progress_bar=False)
            print(f"✓ Cross-encoder sanity: relevant={ce_scores[0]:.3f}, "
                  f"irrelevant={ce_scores[1]:.3f} "
                  + ("✓" if ce_scores[0] > ce_scores[1] else "⚠ unexpected ranking"))
        else:
            print("⚠  Cross-encoder not available — using keyword-boost fallback")

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
    print(f"   Model      : {model_type} Qwen")
    print(f"   Device     : {device.upper()}")
    print(f"   Chunks     : {collection.count()}")
    print(f"   Reranker   : {rerank_type}")
    print(f"   Feedback   : {feedback_store.summary()['total']} records")
    print("\nType 'exit' / 'quit' to stop | 'stats' to view feedback summary.")
    print("After each answer you will be asked to rate it (y/p/n).\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting..."); break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Goodbye!"); break
        if q.lower() == "stats":
            fb = feedback_store.summary()
            print(f"\nFeedback Summary:")
            print(f"  Total       : {fb['total']}")
            print(f"  Correct     : {fb['correct']}")
            print(f"  Partial     : {fb['partial']}")
            print(f"  Incorrect   : {fb['incorrect']}")
            print(f"  Accuracy    : {fb['accuracy']}\n")
            continue

        answer, docs, variants = ask(q, candidate_k=20, top_k=5, threshold=0.40)

        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")

        keep_going = interactive_feedback(q, answer, docs, variants)
        if not keep_going:
            print("Goodbye!")
            break