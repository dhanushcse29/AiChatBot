from huggingface_hub import hf_hub_download

# Login first if private model (optional)
# from huggingface_hub import login
# login()

# Download the model
model_path = hf_hub_download(
    repo_id="Qwen/Qwen-2.7B",
    filename="qwen-2.7b.Q4_K_M.gguf"
)

print("Downloaded model to:", model_path)
