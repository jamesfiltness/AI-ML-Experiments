import json
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report
from huggingface_hub import login

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
BASE_MODEL = "meta-llama/Llama-2-7b-hf"
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ["HF_REPO"]
DATA_PATH = "../data/training_data.jsonl"

INTENTS = [
    "account_update",
    "billing_query",
    "direct_debit_change",
    "drainage_issue",
    "general_enquiry",
    "hardship_support",
    "meter_query",
    "meter_reading_submission",
    "moving_home",
    "planned_outage_enquiry",
    "report_leak",
    "report_low_pressure",
    "report_no_water",
    "water_quality_complaint",
]

login(token=HF_TOKEN)

# -------------------------------------------------------------------
# Load and format dataset (same split as training)
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
test_dataset = split["test"]

print(f"Evaluating on {len(test_dataset)} examples...")

# -------------------------------------------------------------------
# Load base model + adapters from HuggingFace
# -------------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    token=HF_TOKEN,
)

model = PeftModel.from_pretrained(base, HF_REPO, token=HF_TOKEN)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------------------------
# Run evaluation
# -------------------------------------------------------------------
true_labels = []
predicted_labels = []

for i, example in enumerate(test_dataset):
    message = example["text"].split("### Intent:")[0].replace("### Message:", "").strip()
    true_intent = example["text"].split("### Intent:")[-1].strip()

    prompt = f"### Message:\n{message}\n\n### Intent:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = decoded.split("### Intent:")[-1].strip().lower()

    # Match against known intents — longest first to avoid partial matches
    # e.g. maps "account_update_general_enquiry:" back to "account_update"
    predicted = "general_enquiry"
    for intent in sorted(INTENTS, key=len, reverse=True):
        if generated.startswith(intent.lower()):
            predicted = intent
            break

    true_labels.append(true_intent)
    predicted_labels.append(predicted)

    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(test_dataset)} done...")

print("\n--- Results ---\n")
print(classification_report(true_labels, predicted_labels))
