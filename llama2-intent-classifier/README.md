# Water Utility Intent Classifier

https://github.com/user-attachments/assets/e119e224-557e-44c8-8e67-e4d27be2e55a

A fine-tuned Llama 2 7B model for classifying customer service messages from water utility customers into one of 14 intent categories. Trained using QLoRA for efficient fine-tuning on a single GPU.

## What it does

Given a customer message like:

> _"Hi, my water has come out brown this morning, is it safe to drink?"_

The model returns an intent label:

```
water_quality_complaint
```

### Supported intents

| Intent                     | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `report_leak`              | Customer reporting a water leak                     |
| `report_low_pressure`      | Low water pressure complaint                        |
| `report_no_water`          | Complete loss of water supply                       |
| `billing_query`            | Questions or disputes about a bill                  |
| `meter_reading_submission` | Submitting a meter reading                          |
| `meter_query`              | Questions about a water meter                       |
| `direct_debit_change`      | Updating direct debit details                       |
| `account_update`           | Changing account details (name, address, etc.)      |
| `moving_home`              | Moving in or out of a property                      |
| `water_quality_complaint`  | Taste, smell, safety concerns, or discoloured water |
| `planned_outage_enquiry`   | Questions about planned maintenance                 |
| `hardship_support`         | Payment difficulties or financial support           |
| `drainage_issue`           | Blocked drains or sewage problems                   |
| `general_enquiry`          | General questions not covered above                 |

---

## Dataset

The training data (`data/training_data.jsonl`) consists of 1,096 labelled customer messages in conversational format, covering 14 intent types. Generating using a large language model (Claude Opus) Due to lack of real-world data availability.

Messages vary in style — formal and informal, with realistic typos and abbreviations — to reflect real-world customer input.

The dataset was split 90/10 for training and evaluation:

- **Training set:** ~996 examples
- **Test set:** ~111 examples

---

## Why QLoRA?

Full fine-tuning of a 7B parameter model requires significant GPU memory (80GB+) and is expensive. **QLoRA (Quantised Low-Rank Adaptation)** makes this practical on a single consumer or mid-range GPU by:

1. **Quantising the base model to 4-bit** — reducing memory from ~14GB to ~4–5GB
2. **Adding small trainable LoRA adapter layers** — only a fraction of the total parameters are actually updated during training
3. **Dequantising on the fly during the forward pass** — maintaining training quality despite the reduced precision

The result is a fine-tune that costs a fraction of full training in both time and money, with minimal loss in quality for a task like intent classification. Only the adapter weights (~100–300MB) need to be saved and stored, not the full model.

---

## Training

Training was run on [RunPod](https://runpod.io) using a single **NVIDIA A100 (80GB)** GPU.

**Key hyperparameters:**

- Base model: `meta-llama/Llama-2-7b-hf`
- Method: QLoRA (4-bit quantisation + LoRA)
- LoRA rank: 16
- LoRA alpha: 32
- Epochs: 5
- Batch size: 4 (with gradient accumulation)
- Learning rate: 2e-4
- Optimizer: paged AdamW

**Approximate training cost:** ~0.25 on RunPod

Once training completes and adapters are pushed to HuggingFace, run evaluation separately:

```bash
python evaluate.py
```

---

## Model storage

The LoRA adapter weights are stored on [HuggingFace Hub](https://huggingface.co/jamesfiltness/llama2-water-intent) as a private repository. At inference time, the base Llama 2 7B model is loaded and the adapters are applied on top.

---

## Deployment

The model is deployed as a serverless API on [Modal](https://modal.com).

**Example API call:**

```python
import modal

Predictor = modal.Cls.from_name("water-intent-classifier", "Predictor")
result = Predictor().predict.remote("Hey, my water has been completely off since this morning")
print(result)
# report_no_water
```

### Local web app

A simple Flask app is included at `deployment/app.py`. It provides a browser UI that submits messages to the Modal endpoint and polls for the result.

**Setup:**

(I need to learn how to properly set up Python environments on my machine).

```bash
cd llama2-intent-classifier
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install flask modal
```

**Run:**

```bash
.venv/bin/python deployment/app.py
```

Then open [http://localhost:5001](http://localhost:5001).

The app uses a polling pattern — the request returns immediately with a job ID, and the UI polls every 2 seconds until the result is ready. On a cold start (Modal scaling up from zero) the first request may take a couple of minutes due to cold start; subsequent requests within the same session will be faster.

---

## Results

### Run 1 — 1,077 examples, 3 epochs

**Overall accuracy: 75%**

| Intent                   | Precision | Recall | F1   | Support |
| ------------------------ | --------- | ------ | ---- | ------- |
| account_update           | 0.86      | 0.75   | 0.80 | 8       |
| billing_query            | 0.67      | 0.67   | 0.67 | 6       |
| direct_debit_change      | 1.00      | 0.89   | 0.94 | 9       |
| drainage_issue           | 1.00      | 1.00   | 1.00 | 9       |
| general_enquiry          | 0.27      | 0.67   | 0.39 | 9       |
| hardship_support         | 0.75      | 0.27   | 0.40 | 11      |
| meter_query              | 0.88      | 0.88   | 0.88 | 8       |
| meter_reading_submission | 1.00      | 1.00   | 1.00 | 3       |
| moving_home              | 0.83      | 1.00   | 0.91 | 5       |
| planned_outage_enquiry   | 1.00      | 1.00   | 1.00 | 4       |
| report_discoloured_water | 1.00      | 0.50   | 0.67 | 10      |
| report_leak              | 0.64      | 1.00   | 0.78 | 9       |
| report_low_pressure      | 1.00      | 0.71   | 0.83 | 7       |
| report_no_water          | 1.00      | 1.00   | 1.00 | 7       |
| water_quality_complaint  | 0.00      | 0.00   | 0.00 | 3       |

**Observations:**

- 6 intents achieved perfect or near-perfect F1
- `water_quality_complaint` scored 0 — likely confused with `report_discoloured_water` as both involve unusual water
- `hardship_support` and `general_enquiry` had low recall, suggesting the model was misclassifying those messages as other intents
- `report_discoloured_water` had perfect precision but only 50% recall — some examples bleeding into `water_quality_complaint`

**Improvements made for Run 2:**

- Added more training examples for `water_quality_complaint`, `hardship_support`, and `general_enquiry`
- Merged `report_discoloured_water` into `water_quality_complaint` to eliminate confusion between the two similar intents
- Increased training epochs from 3 to 5

---

### Run 2 — 1,107 examples, 5 epochs

**Overall accuracy: 68%**

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| account_update | 1.00 | 0.25 | 0.40 | 8 |
| billing_query | 0.80 | 0.67 | 0.73 | 6 |
| direct_debit_change | 0.88 | 0.88 | 0.88 | 8 |
| drainage_issue | 1.00 | 1.00 | 1.00 | 4 |
| general_enquiry | 0.21 | 0.89 | 0.34 | 9 |
| hardship_support | 0.80 | 0.40 | 0.53 | 10 |
| meter_query | 0.86 | 0.92 | 0.89 | 13 |
| meter_reading_submission | 1.00 | 0.50 | 0.67 | 6 |
| moving_home | 1.00 | 0.60 | 0.75 | 5 |
| planned_outage_enquiry | 1.00 | 1.00 | 1.00 | 3 |
| report_leak | 1.00 | 0.92 | 0.96 | 13 |
| report_low_pressure | 1.00 | 0.75 | 0.86 | 4 |
| report_no_water | 1.00 | 0.38 | 0.55 | 8 |
| water_quality_complaint | 1.00 | 0.57 | 0.73 | 14 |

**Observations:**
- Overall accuracy dropped from 75% to 68% despite some individual improvements
- `water_quality_complaint` improved dramatically from 0.00 to 0.73 — the intent merge worked
- `report_leak` improved significantly from 0.78 to 0.96
- `general_enquiry` precision collapsed to 0.21 — the model is predicting it far too often, overcorrecting from the additional training examples added
- This overcorrection caused regressions in `account_update` (0.80 → 0.40), `report_no_water` (1.00 → 0.55) and `meter_reading_submission` (1.00 → 0.67), as those messages are being incorrectly classified as `general_enquiry`

**Improvements made for Run 3:**
- Reduced `general_enquiry` training examples to prevent overcorrection
- Ensured `general_enquiry` examples are clearly distinct from other intents

---

### Run 3 — 1,096 examples, 5 epochs

**Overall accuracy: 79%**

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|----|---------|
| account_update | 1.00 | 0.86 | 0.92 | 7 |
| billing_query | 0.80 | 0.36 | 0.50 | 11 |
| direct_debit_change | 0.88 | 0.88 | 0.88 | 8 |
| drainage_issue | 1.00 | 1.00 | 1.00 | 11 |
| general_enquiry | 0.20 | 1.00 | 0.33 | 5 |
| hardship_support | 0.88 | 0.64 | 0.74 | 11 |
| meter_query | 1.00 | 0.71 | 0.83 | 7 |
| meter_reading_submission | 1.00 | 1.00 | 1.00 | 1 |
| moving_home | 1.00 | 0.60 | 0.75 | 5 |
| planned_outage_enquiry | 1.00 | 0.89 | 0.94 | 9 |
| report_leak | 1.00 | 0.89 | 0.94 | 9 |
| report_low_pressure | 1.00 | 0.57 | 0.73 | 7 |
| report_no_water | 1.00 | 1.00 | 1.00 | 7 |
| water_quality_complaint | 1.00 | 0.92 | 0.96 | 12 |

**Observations:**
- Best overall accuracy so far at 79%, up from 68% in Run 2 and 75% in Run 1
- `water_quality_complaint` now at 0.96 F1 — a strong result following the intent merge
- `account_update` and `report_no_water` recovered from Run 2 regressions
- `general_enquiry` precision remains very low at 0.20 — still being predicted too broadly despite reducing examples
- `billing_query` recall dropped to 0.36 — likely being confused with `general_enquiry`

---

## Project structure

```
├── data/
│   └── training_data.jsonl
├── training/
│   └── evaluate.py
|   └── finetune.py
├── deployment/
│   └── predict.py
└── README.md
```

---

## Stack

- **Fine-tuning:** [HuggingFace Transformers](https://huggingface.co/docs/transformers), [PEFT](https://huggingface.co/docs/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Training infrastructure:** [RunPod](https://runpod.io)
- **Model registry:** [HuggingFace Hub](https://huggingface.co)
- **Serving:** [Modal](https://modal.com)
