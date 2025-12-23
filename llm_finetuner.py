import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


class LLMFineTuner:
    """
    LLM Fine-Tuning Module
    Can be plugged into any AI Chatbot project
    """

    def __init__(
        self,
        model_name: str,
        dataset_path: str,
        output_dir: str = "./finetuned_model",
        use_qlora: bool = True
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.use_qlora = use_qlora

        self.tokenizer = None
        self.model = None
        self.dataset = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name
            )

    def apply_lora(self):
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def load_dataset(self):
        self.dataset = load_dataset("json", data_files=self.dataset_path)

        def format_prompt(example):
            return {
                "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            }

        self.dataset = self.dataset.map(format_prompt)


    def train(self, epochs=3):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=epochs,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_steps=500,
            report_to="none"
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            tokenizer=self.tokenizer,
            args=training_args
        )

        trainer.train()

    def save(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
