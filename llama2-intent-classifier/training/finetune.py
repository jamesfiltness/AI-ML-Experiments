import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from huggingface_hub import login
from sklearn.metrics import classification_report
import torch

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]          # e.g. "yourname/llama2-water-intent"
DATA_PATH = "../data/training_data.jsonl"
OUTPUT_DIR = "./output"

login(token=HF_TOKEN)

# -------------------------------------------------------------------
# Load and format dataset
# -------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def format_example(example):
    message = example["messages"][0]["content"]
    intent = example["messages"][1]["content"]
    return {"text": f"### Message:\n{message}\n\n### Intent:\n{intent}"}

raw = load_jsonl(DATA_PATH)
formatted = [format_example(ex) for ex in raw]

dataset = Dataset.from_list(formatted)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
test_dataset = split["test"]

print(f"Train: {len(train_dataset)} | Test: {len(test_dataset)}")

# -------------------------------------------------------------------
# Load base model in 4-bit (QLoRA)
# -------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# -------------------------------------------------------------------
# LoRA config
# -------------------------------------------------------------------
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=256,
)

trainer.train()

# -------------------------------------------------------------------
# Save and push adapters to HuggingFace Hub
# -------------------------------------------------------------------
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

model.push_to_hub(HF_REPO, token=HF_TOKEN)
tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)

print(f"Adapters pushed to: https://huggingface.co/{HF_REPO}")

# -------------------------------------------------------------------
# Evaluation — accuracy and F1 per intent
# -------------------------------------------------------------------
print("\nRunning evaluation on test set...")

model.eval()
true_labels = []
predicted_labels = []

for example in test_dataset:
    message = example["text"].split("### Intent:")[0].replace("### Message:", "").strip()
    true_intent = example["text"].split("### Intent:")[-1].strip()

    prompt = f"### Message:\n{message}\n\n### Intent:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted = decoded.split("### Intent:")[-1].strip().split()[0]

    true_labels.append(true_intent)
    predicted_labels.append(predicted)

print(classification_report(true_labels, predicted_labels))
