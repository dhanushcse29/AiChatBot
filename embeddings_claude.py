# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["HF_HOME"]               = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]    = "./hf_cache"

# ================= IMPORTS =================
import uuid
import sys
import re
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

MARKDOWN_DIR         = BASE_DIR / "data" / "markdowns"
CHROMA_DIR           = str(BASE_DIR / "chroma_db")
COLLECTION_NAME      = "pdf_markdown_embeddings_new"
EMBEDDING_MODEL_PATH = str(BASE_DIR / "models" / "BAAI-bge-large")

# ── Chunking settings ─────────────────────────────────────────────────────────
#
#  KEY IMPROVEMENT over the original character-level chunker:
#  ──────────────────────────────────────────────────────────
#  The original chunker cut every 800 *characters* regardless of sentence
#  boundaries.  This meant the last sentence of a chunk was often truncated
#  mid-way, and the first sentence of the next chunk started with a sentence
#  fragment.  The cross-encoder then saw malformed snippets, which degraded
#  reranking quality.
#
#  The new chunker:
#  1. Splits the document into sentences first (regex-based, no NLTK needed).
#  2. Groups sentences greedily until the chunk would exceed MAX_CHARS.
#  3. Overlaps by carrying the last OVERLAP_SENTENCES sentences into the
#     next chunk — so context isn't lost at boundaries.
#  4. Skips "noise" lines (page numbers, standalone digits, very short lines)
#     before chunking.
#
MAX_CHARS         = 900    # max characters per chunk (↑ from 800 gives more context)
OVERLAP_SENTENCES = 3      # how many ending sentences to carry into next chunk
MIN_CHUNK_WORDS   = 30     # discard chunks shorter than this (noise reduction)
EMBED_BATCH_SIZE  = 256

# =========================
# DEVICE
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Using device: {DEVICE}")

# =========================
# EMBEDDING MODEL
# =========================
print("🔹 Loading embedding model...")
embedder = SentenceTransformer(
    EMBEDDING_MODEL_PATH,
    device=DEVICE,
    local_files_only=True
)
print("✅ Embedding model loaded")

# =========================
# CHROMADB — SAFE COSINE INIT (unchanged from original)
# =========================

def get_cosine_collection(
    chroma_client: chromadb.PersistentClient,
    name: str,
) -> chromadb.Collection:

    existing_names = [c.name for c in chroma_client.list_collections()]

    if name in existing_names:
        col   = chroma_client.get_collection(name)
        space = (col.metadata or {}).get("hnsw:space", "l2")
        if space != "cosine":
            print(f"⚠️  Collection '{name}' uses metric='{space}'. "
                  f"Deleting and recreating with cosine metric...")
            chroma_client.delete_collection(name)
        else:
            print(f"✅ Collection '{name}' already uses cosine — reusing.")
            return col

    col = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"✅ Created collection '{name}' with cosine metric.")
    return col


client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = get_cosine_collection(client, COLLECTION_NAME)


# =========================
# IMPROVED CHUNKER
# =========================

# Sentence boundary pattern: split after . ! ? when followed by whitespace + capital
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\(])')

# Noise line patterns to strip before chunking
_NOISE_PATTERNS = [
    re.compile(r'^\s*\d+\s*$'),                      # standalone page numbers
    re.compile(r'^[-=_*]{3,}\s*$'),                   # decorative separator lines
    re.compile(r'page intentionally left blank', re.I),
    re.compile(r'continued on next page', re.I),
]

def _is_noise(line: str) -> bool:
    return any(p.search(line) for p in _NOISE_PATTERNS)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple but robust regex."""
    raw = _SENT_SPLIT.split(text)
    # Further split on newlines that are likely paragraph breaks
    sentences = []
    for s in raw:
        for part in re.split(r'\n{2,}', s):
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def chunk_markdown_sentences(md_path: Path) -> List[str]:
    """
    Sentence-boundary-aware chunker.

    Algorithm:
    1. Load and clean the file (strip noise lines).
    2. Split into sentences.
    3. Greedily accumulate sentences until adding the next would exceed
       MAX_CHARS; emit the current chunk.
    4. Start the next chunk with the last OVERLAP_SENTENCES from the
       previous chunk (sliding window overlap).
    5. Discard chunks shorter than MIN_CHUNK_WORDS.
    """
    with open(md_path, "r", encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    # Strip noise lines
    cleaned_lines = [l for l in raw_lines if not _is_noise(l)]
    text = "".join(cleaned_lines)

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks   : List[str]  = []
    window   : List[str]  = []   # current accumulation buffer
    char_len : int        = 0

    for sent in sentences:
        sent_len = len(sent)

        # If a single sentence is already longer than MAX_CHARS, split it
        # character-wise (rare but can happen with long tables or URLs)
        if sent_len > MAX_CHARS:
            # flush current window first
            if window and char_len >= MIN_CHUNK_WORDS * 5:
                chunks.append(" ".join(window).strip())
            # hard-split the oversized sentence
            for i in range(0, sent_len, MAX_CHARS - 50):
                sub = sent[i : i + MAX_CHARS - 50].strip()
                if sub:
                    chunks.append(sub)
            window, char_len = [], 0
            continue

        if char_len + sent_len > MAX_CHARS and window:
            # Emit current chunk
            chunk_text = " ".join(window).strip()
            if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                chunks.append(chunk_text)
            # Overlap: carry last OVERLAP_SENTENCES into next chunk
            window   = window[-OVERLAP_SENTENCES:] if len(window) > OVERLAP_SENTENCES else window[:]
            char_len = sum(len(s) for s in window)

        window.append(sent)
        char_len += sent_len

    # Flush remaining
    if window:
        chunk_text = " ".join(window).strip()
        if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append(chunk_text)

    return chunks


# =========================
# UTILITIES
# =========================

def load_markdown_files(md_dir: Path) -> List[Path]:
    if not md_dir.exists():
        print(f"❌ MARKDOWN_DIR not found: {md_dir.resolve()}")
        sys.exit(1)
    files = list(md_dir.rglob("*.md"))
    print(f"📄 Found {len(files)} markdown files in {md_dir.resolve()}")
    return files


def validate_batch_norms(vecs: np.ndarray, batch_start: int) -> None:
    norms = np.linalg.norm(vecs, axis=1)
    bad   = np.where((norms < 0.98) | (norms > 1.02))[0]
    if bad.size:
        raise ValueError(
            f"Batch starting at chunk {batch_start}: "
            f"{bad.size} vectors NOT unit-normalised. "
            f"Norms sample: {norms[bad[:5]].tolist()}. "
            f"Ensure normalize_embeddings=True."
        )


# =========================
# INGEST
# =========================

def ingest_markdown() -> None:
    print("\n=== DEBUG INFO ===")
    print(f"CWD           : {os.getcwd()}")
    print(f"MARKDOWN_DIR  : {MARKDOWN_DIR.resolve()} — exists={MARKDOWN_DIR.exists()}")
    print(f"CHROMA_DIR    : {CHROMA_DIR}")
    print(f"Collection    : {COLLECTION_NAME}")
    print(f"Metric        : {(collection.metadata or {}).get('hnsw:space', 'UNKNOWN')}")
    print(f"Chunker       : sentence-boundary (max={MAX_CHARS} chars, "
          f"overlap={OVERLAP_SENTENCES} sentences, min={MIN_CHUNK_WORDS} words)")
    print("==================\n")

    md_files = load_markdown_files(MARKDOWN_DIR)
    if not md_files:
        print("❌ No markdown files found — exiting.")
        return

    total_stored = 0

    for md_path in tqdm(md_files, desc="📄 Markdown files"):
        print(f"\n✂️  Chunking: {md_path.name}")
        chunks = chunk_markdown_sentences(md_path)

        if not chunks:
            print(f"⚠️  No chunks produced for {md_path.name} — skipping.")
            continue

        print(f"🧩 {len(chunks)} chunks produced")

        all_ids        = []
        all_embeddings = []
        all_metadatas  = []

        for batch_start in tqdm(
            range(0, len(chunks), EMBED_BATCH_SIZE),
            desc="🔹 Embedding",
            leave=False,
        ):
            batch = chunks[batch_start : batch_start + EMBED_BATCH_SIZE]

            vecs = embedder.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            validate_batch_norms(vecs, batch_start)

            all_embeddings.extend(vecs.tolist())
            all_ids.extend(str(uuid.uuid4()) for _ in batch)
            all_metadatas.extend(
                {"source": md_path.name, "chunk_index": batch_start + j}
                for j in range(len(batch))
            )

        collection.add(
            ids        = all_ids,
            documents  = chunks,
            embeddings = all_embeddings,
            metadatas  = all_metadatas,
        )

        total_stored += len(all_ids)
        print(f"✅ Stored {len(all_ids)} chunks from {md_path.name}")

    print(f"\n🎉 Ingestion complete — {total_stored} chunks stored total.")
    print(f"   ChromaDB collection count: {collection.count()}")

    _post_ingest_validation()


# =========================
# POST-INGESTION VALIDATION  (unchanged — your original was solid)
# =========================

def _post_ingest_validation() -> None:
    print("\n=== POST-INGESTION VALIDATION ===")

    count = collection.count()
    assert count > 0, "Collection is empty after ingestion!"
    print(f"✅ Count         : {count}")

    space = (collection.metadata or {}).get("hnsw:space", "l2")
    assert space == "cosine", f"Wrong metric: '{space}' (expected 'cosine')"
    print(f"✅ Metric        : {space}")

    sample = collection.get(limit=10, include=["embeddings", "documents"])
    norms  = [np.linalg.norm(e) for e in sample["embeddings"]]
    bad    = [n for n in norms if not (0.97 < n < 1.03)]
    assert not bad, f"Non-unit vectors detected! Bad norms: {bad}"
    print(f"✅ Vector norms  : {[f'{n:.3f}' for n in norms]}")

    first_doc = sample["documents"][0]
    q_vec = embedder.encode(
        [first_doc[:200]], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    res  = collection.query(query_embeddings=q_vec, n_results=1,
                            include=["documents", "distances"])
    dist = res["distances"][0][0]
    sim  = 1.0 - (dist / 2.0)
    print(f"✅ Self-retrieval: dist={dist:.4f}  similarity={sim:.4f}")

    if sim < 0.90:
        print("⚠️  WARNING: self-retrieval similarity is unexpectedly low (<0.90).")
    else:
        print("✅ Self-retrieval sanity check PASSED")

    # NEW — sample a few chunks and show them so you can visually verify
    # they don't end mid-sentence
    print("\n── Sample chunk quality check ──")
    for i, doc in enumerate(sample["documents"][:3], 1):
        words = doc.split()
        last_char = doc.strip()[-1] if doc.strip() else ""
        ends_cleanly = last_char in ".!?:)" or doc.strip()[-1].isalnum()
        print(f"  Chunk {i}: {len(words)} words | ends='{last_char}' | "
              f"{'✅ clean' if ends_cleanly else '⚠ may be truncated'}")
        print(f"    Preview: {doc[:100]}...")

    print("=== VALIDATION COMPLETE ===\n")


# =========================
# ENTRY POINT
# =========================

if __name__ == "__main__":
    ingest_markdown()