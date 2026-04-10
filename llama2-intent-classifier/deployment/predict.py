import os
import modal

app = modal.App("water-intent-classifier")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "torch",
    )
)

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


@app.cls(
    image=image,
    gpu="A100",
    secrets=[modal.Secret.from_name("huggingface")],
)
class Predictor:
    @modal.enter()
    def setup(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        import torch

        BASE_MODEL = "meta-llama/Llama-2-7b-hf"
        ADAPTER_REPO = os.environ["HF_REPO"]    # e.g. "yourname/llama2-water-intent"
        HF_TOKEN = os.environ["HF_TOKEN"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto",
            token=HF_TOKEN,
        )

        self.model = PeftModel.from_pretrained(base, ADAPTER_REPO, token=HF_TOKEN)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @modal.method()
    def predict(self, message: str) -> str:
        import torch

        prompt = f"### Message:\n{message}\n\n### Intent:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted = decoded.split("### Intent:")[-1].strip().split()[0]

        if predicted in INTENTS:
            return predicted

        for intent in INTENTS:
            if intent in decoded:
                return intent

        return "general_enquiry"


# --- For testing locally via: modal run deployment/predict.py --message "your message here"
@app.local_entrypoint()
def main(message: str = "My water has come out brown this morning"):
    predictor = Predictor()
    result = predictor.predict.remote(message)
    print(f"Intent: {result}")
