from llm_finetuner import LLMFineTuner

finetuner = LLMFineTuner(
    model_name="mistralai/Mistral-7B-Instruct-v0.2",  # GPU
    dataset_path="train.json",
    output_dir="./chatbot_lora_model",
    use_qlora=True
)

finetuner.load_model()
finetuner.apply_lora()
finetuner.load_dataset()
finetuner.train(epochs=3)
finetuner.save()
