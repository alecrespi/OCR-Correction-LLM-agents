from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    infer_device,
    BitsAndBytesConfig
)
from huggingface_hub import login
import numpy as np
import torch
import json
from tqdm import tqdm
import gc
import os

def get_latest_attempt():
    filelist = os.listdir(DESTINATION_PATH)
    attempts = []
    for a in filelist:
        if a.endswith(".json") and a.startswith("attempt."):
            try:
                attempts.append(int(a.split(".")[-2]))
            except ValueError:
                pass
    return max(attempts) if attempts else 0


class InferenceHandler:
    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device if device is None else device

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def run_batch(self, texts, batch_size=4):
        all_results = []

        for i in range(0, len(texts), batch_size):
            print(f"Processing batch {i//batch_size + 1} / {(len(texts) + batch_size - 1)//batch_size} ...")
            batch_input = texts[i:i+batch_size]
            prompts = np.array([(key, PROMPT.replace("[INPUT]",text)) for key, text in batch_input])
            
            # tokenize input
            inputs = self.tokenizer(
                prompts[:, 1].tolist(), 
                return_tensors="pt",
                padding=False,
                # truncation=False,
                max_length=int(2e12), # expected max size
            ).to(self.device)
            
            # generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = MAX_NEW_TOKENS,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for i, (prompt, result) in enumerate(zip(prompts[:, 1], batch_results)):
                key = batch_input[i][0]  
                corrected = result.replace(prompt, "").strip()
                all_results.append((key, corrected))
        
        return {key: res for key, res in all_results}


# PARAMETERS
HF_LOGIN_TOKEN = os.environ.get("HF_LOGIN_TOKEN", "")
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", 60 * 19))  # right before the "espresso" task timeout
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2-1.5B-Instruct")
MODEL_SHORT_NAME = os.environ.get("MODEL_SHORT_NAME", MODEL_NAME)
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 4))
PROMPT = os.environ.get("PROMPT", "Correct OCR scanning errors in the following text. Fix ONLY character recognition mistakes, not content.\n[INPUT]\nOutput:\n")
DATASET_LANG = os.environ.get("DATASET_LANG", "ENG")
OCRED_PATH = os.environ.get("OCRED_PATH", "./DATASETS/" + DATASET_LANG + "/ocred.json")
CLEANED_PATH = os.environ.get("CLEANED_PATH", "./DATASETS/" + DATASET_LANG + "/cleaned.json")
DESTINATION_PATH = os.environ.get("DESTINATION_PATH", "./CORRECTIONS/" + DATASET_LANG + "/" + MODEL_SHORT_NAME + "/")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", "/tmp/huggingface_cache")
MODEL_MODE = os.environ.get("MODEL_MODE", "text-generation")  # "text-generation" or "text2text-generation"
DATASET_START = int(os.environ.get("DATASET_START", 0))
DATASET_END = int(os.environ.get("DATASET_END", -1))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 256))
DEVICE = os.environ.get("DEVICE", "auto")

# PROGRAM
print("START PROGRAM...")
print("Logging into HuggingFace Hub...")
if HF_LOGIN_TOKEN:
    login(HF_LOGIN_TOKEN)
    print("Logged in!")
else:
    print("No HF_LOGIN_TOKEN provided, proceeding without login...")
os.makedirs(os.path.dirname(DESTINATION_PATH), exist_ok=True)
destination_file = os.path.join(DESTINATION_PATH, "attempt." + str(get_latest_attempt()) + ".json")

print("Inferred device: " + DEVICE)
print("Loading the agent: " + MODEL_SHORT_NAME)

model = None
tokenizer = None
# 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

if MODEL_MODE == "text2text-generation":
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        cache_dir=HF_HUB_CACHE,
        device_map="auto",
        token=HF_LOGIN_TOKEN if HF_LOGIN_TOKEN else None,
        trust_remote_code=True
    )
else: 
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        cache_dir=HF_HUB_CACHE,
        device_map="auto",
        token=HF_LOGIN_TOKEN if HF_LOGIN_TOKEN else None,
        trust_remote_code=True
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_LOGIN_TOKEN if HF_LOGIN_TOKEN else None
)
print("Agent loaded! Eval: ")
print(model.eval())

# loading dataset
ocred = {}
cleaned = {}
with open(OCRED_PATH, 'r') as f:
    ocred = list(json.load(f).items())

# defining inference
DATASET_END = len(ocred) if DATASET_END == -1 else int(DATASET_END)

print(f"Running inference on {DATASET_END - DATASET_START} samples, from {DATASET_START} to {DATASET_END}, in batches of {BATCH_SIZE}...")
inferer = InferenceHandler(model, tokenizer, device=DEVICE)
results = inferer.run_batch(ocred[DATASET_START:DATASET_END], batch_size=BATCH_SIZE)
print(f"Generated {len(results)} results.")

# saving results
with open(destination_file, 'w+') as f:
    json.dump({key: result for key, result in results.items()}, f, indent=4)

print("Results saved to: " + destination_file)