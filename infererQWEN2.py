### The fucking program (fucking finally, it's fucking 01:35 AM for fuck sake)
from transformers import AutoModelForCausalLM, AutoTokenizer, infer_device
import numpy as np
import torch
import json
from tqdm import tqdm
import gc
import os

# PARAMETERS
# GENERATION_MODE = "text-generation"
MODEL_NAME = "Qwen/Qwen2-1.5B-Instruct"
SHORT_NAME = "Qwen2"

BATCH_SIZE = 4

LEADING_PROMPT = "Correct OCR scanning errors in the following text. Fix ONLY character recognition mistakes, not content.\n"
TRAILING_PROMPT = "\nProvide only the corrected text with minimal character-level fixes: "

LANG = "ENG"
OCRED_PATH = "./DATASETS/" + LANG + "/ocred.json"
CLEANED_PATH = "./DATASETS/" + LANG + "/cleaned.json"

DESTINATION_PATH = "./CORRECTIONS/" + LANG + "/" + SHORT_NAME + "/attempt.json"
os.makedirs(os.path.dirname(DESTINATION_PATH))

device = infer_device()
print("Inferred device: " + device if device is not None else "[No device found]")
print("Loading the agent: " + MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Agent loaded! Eval: ")
print(model.eval())

# loading dataset
ocred = {}
cleaned = {}
with open(OCRED_PATH, 'r') as f:
    ocred = json.load(f)
    ocred = np.array(list(ocred.values()))

# with open(CLEANED_PATH, 'r') as f:
#     cleaned = json.load(f)
#     cleaned = np.array(list(cleaned.values()))

# ocred[list(ocred.keys())[:5]]
# list(cleaned)

START = 0
LIMIT = len(ocred)

class InferenceHandler:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def run_batch(self, texts, batch_size=4):
        all_results = []
        
        for i in range(0, len(texts), batch_size):
            batch_input = texts[i:i+batch_size]
            prompts = [f"{LEADING_PROMPT}{text}{TRAILING_PROMPT}" for text in batch_input]
            
            # tokenize input
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                # truncation=True,
                max_length=512,
            )
            
            # generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    use_cache=False  # Save memory
                )
            
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for prompt, result in zip(prompts, batch_results):
                corrected = result.replace(prompt, "").strip()
                all_results.append(corrected)
            
            # # cleanup ...
            # del inputs, outputs
            # torch.cuda.empty_cache()
            # gc.collect()
        
        return all_results

inferer = InferenceHandler(model, tokenizer)
results = inferer.run_batch(ocred[START:LIMIT], batch_size=BATCH_SIZE)

# saving results
with open(DESTINATION_PATH, 'w') as f:
    json.dump({str(i+START): res for i, res in enumerate(results)}, f, indent=4)
