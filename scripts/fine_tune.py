import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# Load configuration
config_path = "../config/model_config.json"
with open(config_path, "r") as f:
    config = json.load(f)

# Extract model and training arguments from config
model_name = config["model_name"]
output_dir = config["output_dir"]
dataset_path = config["dataset_path"]
max_length = config["max_length"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
num_epochs = config["num_epochs"]
evaluation_strategy = config["evaluation_strategy"]
save_steps = config["save_steps"]
logging_dir = config["logging_dir"]
gradient_accumulation_steps = config["gradient_accumulation_steps"]
gradient_checkpointing = config["gradient_checkpointing"]
weight_decay = config["weight_decay"]
save_total_limit = config["save_total_limit"]
fp16 = config["fp16"]
max_grad_norm = config["max_grad_norm"]

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add padding token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    offload_folder="offload",
    offload_state_dict=True
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, lora_config)

# Ensure gradients are enabled only for floating-point parameters
for name, param in model.named_parameters():
    if param.dtype in [torch.float32, torch.float16, torch.bfloat16]:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Load dataset
print("Loading dataset...")
dataset = load_dataset("json", data_files={"train": dataset_path})

def tokenize_function(example):
    tokens = tokenizer(
        example["instruction"],
        text_pair=example["response"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    tokens["labels"] = tokens["input_ids"].copy()  # Set labels to input_ids for causal LM
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy=evaluation_strategy,
    save_steps=save_steps,
    logging_dir=logging_dir,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=gradient_checkpointing,
    num_train_epochs=num_epochs,
    weight_decay=weight_decay,
    save_total_limit=save_total_limit,
    fp16=fp16,
    max_grad_norm=max_grad_norm
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Fine-tune the model
print("Starting training...")
trainer.train(resume_from_checkpoint="../models/fine_tuned")
print("Training complete. Model saved to", output_dir)