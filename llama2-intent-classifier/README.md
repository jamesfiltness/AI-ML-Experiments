# Water Utility Intent Classifier

A fine-tuned Llama 2 7B model for classifying customer service messages from water utility customers into one of 15 intent categories. Trained using QLoRA for efficient fine-tuning on a single GPU.

## What it does

Given a customer message like:

> *"Hi, my water has come out brown this morning, is it safe to drink?"*

The model returns an intent label:

```
report_discoloured_water
```

### Supported intents

| Intent | Description |
|--------|-------------|
| `report_leak` | Customer reporting a water leak |
| `report_low_pressure` | Low water pressure complaint |
| `report_discoloured_water` | Discoloured or unusual looking water |
| `report_no_water` | Complete loss of water supply |
| `billing_query` | Questions or disputes about a bill |
| `meter_reading_submission` | Submitting a meter reading |
| `meter_query` | Questions about a water meter |
| `direct_debit_change` | Updating direct debit details |
| `account_update` | Changing account details (name, address, etc.) |
| `moving_home` | Moving in or out of a property |
| `water_quality_complaint` | Taste, smell, or safety concerns |
| `planned_outage_enquiry` | Questions about planned maintenance |
| `hardship_support` | Payment difficulties or financial support |
| `drainage_issue` | Blocked drains or sewage problems |
| `general_enquiry` | General questions not covered above |

---

## Dataset

The training data (`data/training_data.jsonl`) consists of 1,077 labelled customer messages in conversational format, covering all 15 intent types.

Messages vary in style — formal and informal, with realistic typos and abbreviations — to reflect real-world customer input.

The dataset was split 90/10 for training and evaluation:
- **Training set:** ~969 examples
- **Test set:** ~108 examples

---

## Why QLoRA?

Full fine-tuning of a 7B parameter model requires significant GPU memory (80GB+) and is expensive. **QLoRA (Quantised Low-Rank Adaptation)** makes this practical on a single consumer or mid-range GPU by:

1. **Quantising the base model to 4-bit** — reducing memory from ~14GB to ~4–5GB
2. **Adding small trainable LoRA adapter layers** — only a fraction of the total parameters are actually updated during training
3. **Dequantising on the fly during the forward pass** — maintaining training quality despite the reduced precision

The result is a fine-tune that costs a fraction of full training in both time and money, with minimal loss in quality for a task like intent classification. Only the adapter weights (~100–300MB) need to be saved and stored, not the full model.

---

## Training

Training was run on [RunPod](https://runpod.io) using a single **NVIDIA A100 40GB** GPU.

**Key hyperparameters:**
- Base model: `meta-llama/Llama-2-7b-hf`
- Method: QLoRA (4-bit quantisation + LoRA)
- LoRA rank: 16
- LoRA alpha: 32
- Epochs: 3
- Batch size: 4 (with gradient accumulation)
- Learning rate: 2e-4
- Optimizer: paged AdamW

**Approximate training cost:** ~$1–2 on RunPod

---

## Model storage

The LoRA adapter weights are stored on [HuggingFace Hub](<your-hf-repo-url>) as a private repository. At inference time, the base Llama 2 7B model is loaded and the adapters are applied on top.

---

## Deployment

The model is deployed as a serverless API on [Replicate](<your-replicate-model-url>).

**Example API call:**

```python
import replicate

output = replicate.run(
    "<your-replicate-model>",
    input={"message": "Hey, my water has been completely off since this morning"}
)

print(output)
# report_no_water
```

---

## Results

| Metric | Score |
|--------|-------|
| Test accuracy | _TBC_ |
| Macro F1 | _TBC_ |

_Confusion matrix and loss curve coming soon._

---

## Project structure

```
├── data/
│   └── training_data.jsonl
├── training/
│   └── finetune.py
├── deployment/
│   └── predict.py
└── README.md
```

---

## Stack

- **Fine-tuning:** [HuggingFace Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Training infrastructure:** [RunPod](https://runpod.io)
- **Model registry:** [HuggingFace Hub](https://huggingface.co)
- **Serving:** [Replicate](https://replicate.com)
