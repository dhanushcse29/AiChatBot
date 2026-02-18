import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"C:\Users\pooji\OneDrive\Desktop\using_chroma-main\models\qwen"
ADAPTER_PATH = r"C:\Users\pooji\OneDrive\Desktop\using_chroma-main\models\qwen_qlora_adapters"

# ---------------- LOAD MODEL ----------------
print("Loading Qwen base model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("✅ Qwen model loaded with fine-tuned adapters\n")

# ---------------- TEST INFERENCE ----------------
def generate_response(question):
    messages = [
        {
            "role": "system",
            "content": "You are a technical assistant specializing in EW systems. Answer questions accurately and concisely."
        },
        {
            "role": "user",
            "content": question
        }
    ]
    
    # Apply Qwen chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
    
    return response.strip()

# ---------------- INTERACTIVE TESTING ----------------
print("Type your questions (type 'exit' to quit):\n")

while True:
    question = input(">>> ")
    if question.lower() in ['exit', 'quit']:
        break
    
    answer = generate_response(question)
    print(f"\n{answer}\n")