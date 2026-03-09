# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# ================= IMPORTS =================
import uuid
import sys
import numpy as np
from pathlib import Path
from typing import List

import torch
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent

MARKDOWN_DIR         = BASE_DIR / "data" / "markdowns"
CHROMA_DIR           = str(BASE_DIR / "chroma_store")
COLLECTION_NAME      = "pdf_markdown_embeddings"
EMBEDDING_MODEL_PATH = str(BASE_DIR / "models" / "BAAI-bge-large")

CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
EMBED_BATCH_SIZE = 256

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
# CHROMADB — SAFE COSINE INIT
# =========================
# ⚠️  get_or_create_collection does NOT update the metric if the collection
#     already exists with a different metric (e.g. default L2).
#     The only safe guarantee is to inspect existing metadata and recreate
#     when the metric doesn't match.

def get_cosine_collection(
    chroma_client: chromadb.PersistentClient,
    name: str,
) -> chromadb.Collection:

    existing_names = [c.name for c in chroma_client.list_collections()]

    if name in existing_names:
        col   = chroma_client.get_collection(name)
        space = (col.metadata or {}).get("hnsw:space", "l2")

        if space != "cosine":
            print(
                f"⚠️  Collection '{name}' exists with metric='{space}'.\n"
                f"   Deleting and recreating with cosine metric..."
            )
            chroma_client.delete_collection(name)
        else:
            print(f"✅ Collection '{name}' already uses cosine — reusing.")
            return col

    col = chroma_client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}   # ← baked in permanently at creation
    )
    print(f"✅ Created collection '{name}' with cosine metric.")
    return col


client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = get_cosine_collection(client, COLLECTION_NAME)

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


def chunk_markdown_stream(md_path: Path) -> List[str]:
    """
    Sliding-window character-level chunker.
    Each chunk is at most CHUNK_SIZE chars; consecutive chunks overlap
    by CHUNK_OVERLAP chars so context is not lost at boundaries.
    """
    chunks  = []
    current = ""

    with open(md_path, "r", encoding="utf-8") as f:
        for line in f:
            current += line
            if len(current) >= CHUNK_SIZE:
                chunks.append(current.strip())
                current = current[-CHUNK_OVERLAP:]   # keep tail for overlap

    if current.strip():
        chunks.append(current.strip())

    return chunks


def validate_batch_norms(vecs: np.ndarray, batch_start: int) -> None:
    """Assert every vector in the batch is unit-normalised (fails fast)."""
    norms = np.linalg.norm(vecs, axis=1)
    bad   = np.where((norms < 0.98) | (norms > 1.02))[0]
    if bad.size:
        raise ValueError(
            f"Batch starting at chunk {batch_start}: "
            f"{bad.size} vectors NOT unit-normalised. "
            f"Norms sample: {norms[bad[:5]].tolist()}. "
            f"Ensure normalize_embeddings=True is set."
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
    print("==================\n")

    md_files = load_markdown_files(MARKDOWN_DIR)
    if not md_files:
        print("❌ No markdown files found — exiting.")
        return

    total_stored = 0

    for md_path in tqdm(md_files, desc="📄 Markdown files"):
        print(f"\n✂️  Chunking: {md_path.name}")
        chunks = chunk_markdown_stream(md_path)

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
                normalize_embeddings=True,   # ← MUST stay True
                convert_to_numpy=True,
                show_progress_bar=False,      # outer tqdm handles display
            )

            # Norm guard — catches silent normalisation failures immediately
            validate_batch_norms(vecs, batch_start)

            all_embeddings.extend(vecs.tolist())
            all_ids.extend(str(uuid.uuid4()) for _ in batch)
            all_metadatas.extend(
                {"source": md_path.name, "chunk_index": batch_start + j}
                for j in range(len(batch))
            )

        # Write entire file's chunks in one call (fewer Chroma round-trips)
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
# POST-INGESTION VALIDATION
# =========================
def _post_ingest_validation() -> None:
    """
    Runs automatically after ingestion.
    Catches the most common silent failures before inference time.
    """
    print("\n=== POST-INGESTION VALIDATION ===")

    # 1. Count
    count = collection.count()
    assert count > 0, "Collection is empty after ingestion!"
    print(f"✅ Count         : {count}")

    # 2. Metric
    space = (collection.metadata or {}).get("hnsw:space", "l2")
    assert space == "cosine", f"Wrong metric: '{space}' (expected 'cosine')"
    print(f"✅ Metric        : {space}")

    # 3. Stored-vector norms
    sample = collection.get(limit=10, include=["embeddings", "documents"])
    norms  = [np.linalg.norm(e) for e in sample["embeddings"]]
    bad    = [n for n in norms if not (0.97 < n < 1.03)]
    assert not bad, f"Non-unit vectors detected in store! Bad norms: {bad}"
    print(f"✅ Vector norms  : {[f'{n:.3f}' for n in norms]}")

    # 4. Self-retrieval smoke test
    #    Re-embed the first stored doc text and query — top result must be itself
    first_doc = sample["documents"][0]
    q_vec = embedder.encode(
        [first_doc[:200]],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    res  = collection.query(
        query_embeddings=q_vec,
        n_results=1,
        include=["documents", "distances"],
    )
    dist = res["distances"][0][0]
    sim  = 1.0 - (dist / 2.0)   # ChromaDB cosine distance ∈ [0, 2] → similarity ∈ [0, 1]

    print(f"✅ Self-retrieval: dist={dist:.4f}  similarity={sim:.4f}")

    if sim < 0.90:
        print(
            "⚠️  WARNING: self-retrieval similarity is unexpectedly low (<0.90).\n"
            "   Likely cause: different embedding model was used in a previous\n"
            "   ingestion run. Verify EMBEDDING_MODEL_PATH and re-ingest."
        )
    else:
        print("✅ Self-retrieval sanity check PASSED")

    print("=== VALIDATION COMPLETE ===\n")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    ingest_markdown()