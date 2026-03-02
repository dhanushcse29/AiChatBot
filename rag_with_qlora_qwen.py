# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# ================= IMPORTS =================
import re
import sys
import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = str(BASE_DIR / "models" / "qwen")
ADAPTER_PATH    = str(BASE_DIR / "models" / "qwen_qlora_adapters")
EMBED_PATH      = str(BASE_DIR / "models" / "all-MiniLM-L6-v2")
CHROMA_DIR      = str(BASE_DIR / "chroma_store")
COLLECTION_NAME = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ================= LOAD EMBEDDER =================
print("Loading embedder (OFFLINE)...")
embedder = SentenceTransformer(EMBED_PATH, device="cuda", local_files_only=True)
print("✅ Embedder loaded")

# ================= LOAD CHROMA =================
print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Collection loaded — {collection.count()} chunks")

# ── Validate distance metric at startup (fail fast)
space = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"   Distance metric: {space}")
if space != "cosine":
    print(
        f"\n⛔  FATAL: Collection metric is '{space}', expected 'cosine'.\n"
        "    Re-run embeddings.py to recreate the collection with the correct metric.\n"
        "    Do NOT continue — retrieval results will be meaningless.\n"
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

# ================= RETRIEVAL =================
def retrieve(query: str, k: int = 5, threshold: float = 0.35,
             verbose: bool = True) -> list[str]:
    """
    Retrieve relevant chunks using cosine similarity.

    ChromaDB cosine distance is in [0, 2]:
        0   → vectors are identical
        1   → vectors are orthogonal
        2   → vectors are opposite

    We convert:
        similarity = 1 - (distance / 2)   →  range [0, 1]

    Threshold of 0.35 means "at least 35% cosine similarity".
    Raise threshold (e.g. 0.50) to be stricter.
    Lower threshold (e.g. 0.20) to be more permissive.
    """
    # Embed query
    q_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    if verbose:
        print(f"Query vector norm: {np.linalg.norm(q_vec[0]):.4f}")

    # Query ChromaDB
    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "distances"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]

    filtered_docs = []
    similarities = []

    for rank, (doc, dist) in enumerate(zip(docs, distances), 1):
        similarity = 1.0 - (dist / 2.0)
        similarities.append(similarity)

        if verbose:
            print(f"{rank:<5} {dist:>8.4f} {similarity:>11.4f}  {doc[:55]}...")

    # ---- HARD GUARD ----
    top_similarity = similarities[0]

    if top_similarity < 0.70:
        print(f"⚠ Weak retrieval (top similarity={top_similarity:.3f})")
        return []

    for doc, sim in zip(docs, similarities):
        if sim >= threshold:
            filtered_docs.append(doc)
    print(f"\n🔎 Retrieved {len(filtered_docs)} / {len(docs)} chunks "
            f"(threshold={threshold})")
    return filtered_docs


# ================= QUESTION TYPE =================
def detect_question_type(question: str) -> str:
    q = question.lower()
    if q.startswith(("what is", "define", "meaning of")):
        return "definition"
    if q.startswith(("explain", "describe", "overview")):
        return "explanation"
    if q.startswith(("how", "procedure", "steps", "process")):
        return "procedure"
    if any(w in q for w in ["safety", "measures", "precautions", "guidelines"]):
        return "list"
    return "general"


# ================= CLEAN CONTEXT =================
def clean_context(context: str) -> str:
    cleaned = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if any(x in lower for x in ["page", "table", "figure",
                                     "intentionally left blank"]):
            continue
        if len(line) < 30:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ================= COMPLETE SENTENCE =================
def ensure_complete_sentence(text: str) -> str:
    text = text.strip()
    if text.endswith((".", "!", "?")):
        return text
    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        return text[: matches[-1].end()].strip()
    return text


# ================= PROMPT BUILDER =================
def build_prompt(context: str, question: str, qtype: str) -> str:
    style_map = {
        "definition":   "Give a short, precise definition in one paragraph.",
        "explanation":  "Provide a clear and complete explanation in 5–7 sentences.",
        "procedure":    "Explain step by step in a clear sequence.",
        "list":         "Answer using clear bullet points.",
        "general":      "Provide a concise, factual explanation.",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are a technical assistant specialising in EW systems.\n\n"
                "Answer using ONLY the context provided.\n\n"
                "Rules:\n"
                f"- {style_map[qtype]}\n"
                "- Do NOT invent information.\n"
                "- If the context does not contain the answer, say exactly:\n"
                "  \"I don't have information on that.\""
            ),
        },
        {
            "role": "user",
            "content": f"### CONTEXT:\n{context}\n\n### QUESTION:\n{question}",
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ================= GENERATION =================
def generate_answer(context: str, question: str) -> str:
    qtype   = detect_question_type(question)
    context = clean_context(context)
    prompt  = build_prompt(context, question, qtype)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Strip any leaked chat template tokens
    if "<|im_start|>assistant" in answer:
        answer = answer.split("<|im_start|>assistant")[-1]
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0]

    answer = ensure_complete_sentence(answer.strip())

    if not answer or len(answer.split()) < 5:
        return "I don't have information on that."

    return answer


# ================= RAG PIPELINE =================
def ask(question: str, threshold: float = 0.35) -> str:
    """
    Full RAG pipeline:
      1. Embed query
      2. Retrieve relevant chunks (cosine similarity ≥ threshold)
      3. Guard against empty / thin context
      4. Generate grounded answer via Qwen
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {question}")
    print('='*60)

    docs = retrieve(question, k=5, threshold=threshold)

    if not docs:
        # ── Optionally retry with a lower threshold before giving up
        print("⚠  No results at threshold={threshold}. Retrying at 0.20...")
        docs = retrieve(question, k=5, threshold=0.20, verbose=False)
        if not docs:
            return "I don't have information on that."

    context = "\n\n".join(docs)

    if len(context.split()) < 40:
        return "I don't have information on that."

    return generate_answer(context, question)


# ================= STARTUP DIAGNOSTIC =================
def run_startup_diagnostic():
    """
    Validates that the stored embeddings are compatible with the
    current embedder and that retrieval is working end-to-end.
    Run once at startup; remove or comment out for production.
    """
    print("\n=== STARTUP DIAGNOSTIC ===")

    # 1. Stored vector shape & norm
    sample = collection.get(limit=5, include=["embeddings", "documents"])
    norms  = [np.linalg.norm(e) for e in sample["embeddings"]]
    print(f"Stored vector dim  : {len(sample['embeddings'][0])}")
    print(f"Stored vector norms: {[f'{n:.3f}' for n in norms]}")
    if not all(0.95 < n < 1.05 for n in norms):
        print("⚠  WARNING: stored vectors are NOT unit-normalised.\n"
              "   Re-run embeddings.py with normalize_embeddings=True.")

    # 2. Self-retrieval smoke test
    first_doc = sample["documents"][0]
    q_vec = embedder.encode([first_doc[:200]], normalize_embeddings=True,
                            convert_to_numpy=True).tolist()
    res   = collection.query(query_embeddings=q_vec, n_results=1,
                             include=["documents", "distances"])
    dist  = res["distances"][0][0]
    sim   = 1.0 - (dist / 2.0)
    print(f"Self-retrieval: dist={dist:.4f}, similarity={sim:.4f}")
    if sim < 0.90:
        print("⚠  WARNING: self-retrieval similarity is low.\n"
              "   Possible embedding-space mismatch between ingestion and inference.")
    else:
        print("✅ Self-retrieval OK — embeddings are compatible.")

    print("=== END DIAGNOSTIC ===\n")


# ================= MAIN =================
if __name__ == "__main__":
    run_startup_diagnostic()

    model_type = "Fine-tuned" if use_finetuned else "Base"
    print(f"\n✅ RAG READY — {model_type} Qwen | Offline | {device.upper()}")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        answer = ask(q)
        print(f"\n{answer}\n")
