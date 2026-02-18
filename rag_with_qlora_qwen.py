import json
import re
import torch
import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"D:/Machine Learning and LLMs/LLMs/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = r"D:/Interns/pdf_extraction_using_chromadb/models/qwen_qlora_adapters"
EMBED_PATH = r"D:/Machine Learning and LLMs/LLMs/all-MiniLM-L6-v2"
CHROMA_DIR = r"D:/Interns/pdf_extraction_using_chromadb/chroma_store"
COLLECTION_NAME = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD EMBEDDER ----------------
print("Loading embedder...")
embedder = SentenceTransformer(EMBED_PATH, device=device)

# ---------------- LOAD FINE-TUNED LLM ----------------
print("Loading fine-tuned Qwen model...")

# Check if adapters exist
use_finetuned = os.path.exists(ADAPTER_PATH)

if use_finetuned:
    print("✓ Fine-tuned adapters found. Loading with QLoRA...")
    
    # Quantization config for 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model in 4-bit
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True  # Required for Qwen
    )
    
    # Load fine-tuned adapters
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ Fine-tuned Qwen model loaded successfully!")
    
else:
    print("⚠ No fine-tuned adapters found. Loading base model...")
    
    # Load base model without quantization
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

# Set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ---------------- LOAD CHROMA ----------------
print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

print(f"Loaded {collection.count()} chunks from ChromaDB")

# ---------------- RETRIEVAL (CHROMA) ----------------
def retrieve(query, k=25):
    """Retrieve top-k relevant chunks from ChromaDB"""
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0]

# ---------------- QUESTION TYPE DETECTION ----------------
def detect_question_type(question: str) -> str:
    """Detect the type of question to customize response style"""
    q = question.lower()

    if q.startswith(("what is", "define", "meaning of")):
        return "definition"
    if q.startswith(("explain", "describe", "overview")):
        return "explanation"
    if q.startswith(("how", "procedure", "steps", "process")):
        return "procedure"
    if any(word in q for word in ["safety", "measures", "precautions", "guidelines"]):
        return "list"
    return "general"

# ---------------- CONTEXT CLEANING ----------------
def clean_context(context: str) -> str:
    """Remove unwanted lines from context"""
    lines = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        # Skip figure/table captions
        if lower.startswith(("figure", "table")) and len(line.split()) < 6:
            continue
        lines.append(line)
    return "\n".join(lines)

# ---------------- SENTENCE COMPLETION ----------------
def ensure_complete_sentence(text: str) -> str:
    """Ensure the response ends with a complete sentence"""
    text = text.strip()
    if text.endswith((".", "!", "?")):
        return text
    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        return text[:matches[-1].end()].strip()
    return text

# ---------------- PROMPT BUILDER (QWEN FORMAT) ----------------
def build_prompt(context: str, question: str, qtype: str) -> str:
    """Build Qwen chat-format prompt with context and question"""
    style_map = {
        "definition": "Give a short, precise definition in one paragraph.",
        "explanation": "Provide a clear and complete explanation in 5–7 sentences.",
        "procedure": "Explain step by step in a clear sequence.",
        "list": "Answer using clear bullet points.",
        "general": "Provide a concise, factual explanation."
    }

    messages = [
        {
            "role": "system",
            "content": f"""You are a technical assistant specializing in EW systems.

Answer using ONLY the context provided.

Rules:
- {style_map[qtype]}
- Do NOT mention chapters, figures, or tables.
- Do NOT invent information.
- If the context does not contain the answer, say exactly: "I don't have information on that."
"""
        },
        {
            "role": "user",
            "content": f"""### CONTEXT:
{context}

### QUESTION:
{question}"""
        }
    ]
    
    # Apply Qwen's chat template
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# ---------------- ANSWER GENERATION ----------------
def generate_answer(context: str, question: str) -> str:
    """Generate answer using fine-tuned Qwen model with retrieved context"""
    qtype = detect_question_type(question)
    context = clean_context(context)
    prompt = build_prompt(context, question, qtype)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in answer:
        answer = answer.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0]
    
    # Clean up
    answer = answer.strip()
    answer = ensure_complete_sentence(answer)

    if not answer or len(answer.split()) < 5:
        return "I don't have information on that."

    return answer

# ---------------- RAG PIPELINE ----------------
def ask(question: str) -> str:
    """Main RAG pipeline: retrieve context and generate answer"""
    docs = retrieve(question)
    if not docs:
        return "I don't have information on that."
    context = "\n\n".join(docs)
    return generate_answer(context, question)

# ---------------- CHAT INTERFACE ----------------
model_type = "Fine-tuned" if use_finetuned else "Base"
print(f"\n✅ RAG READY — {model_type} Qwen Model, ChromaDB Backend, Offline, GPU-Accelerated\n")

while True:
    q = input(">>> ")
    if q.lower() in ["exit", "quit"]:
        break
    print("\n", ask(q), "\n")