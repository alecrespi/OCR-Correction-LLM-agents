from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    infer_device,
    BitsAndBytesConfig,
    # ===== ADDED FOR FINE-TUNING =====
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
    # ===== END ADDED =====
)
from huggingface_hub import login
import numpy as np
import torch
import json
from tqdm import tqdm
import gc
import os
# ===== ADDED FOR FINE-TUNING =====
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from torch.utils.data import Dataset
import random
# ===== END ADDED =====

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


# ===== ADDED FOR FINE-TUNING =====
class OCRCorrectionDataset(Dataset):
    """Custom dataset for OCR correction task"""
    def __init__(self, ocred_texts, cleaned_texts, tokenizer, prompt_template, max_length=512):
        self.ocred_texts = ocred_texts
        self.cleaned_texts = cleaned_texts
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.max_length = max_length
        
    def __len__(self):
        return len(self.ocred_texts)
    
    def __getitem__(self, idx):
        ocred_text = self.ocred_texts[idx]
        cleaned_text = self.cleaned_texts[idx]
        
        # Format the input with prompt
        input_text = self.prompt_template.replace("[INPUT]", ocred_text)
        
        # For causal LM, we need to include the target in the input
        if MODEL_MODE == "text-generation":
            full_text = input_text + cleaned_text
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Create labels (mask the input part, keep only the output part for loss)
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                add_special_tokens=False
            )
            labels = encoding["input_ids"].clone()
            labels[:, :len(input_encoding["input_ids"])] = -100  # Mask input tokens
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": labels.squeeze()
            }
        else:  # text2text-generation
            input_encoding = self.tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            target_encoding = self.tokenizer(
                cleaned_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": input_encoding["input_ids"].squeeze(),
                "attention_mask": input_encoding["attention_mask"].squeeze(),
                "labels": target_encoding["input_ids"].squeeze()
            }


def setup_qlora_config(model):
    """Setup QLoRA configuration"""
    # Determine task type based on model mode
    task_type = TaskType.CAUSAL_LM if MODEL_MODE == "text-generation" else TaskType.SEQ_2_SEQ_LM
    
    # QLoRA configuration
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=task_type,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Get PEFT model
    model = get_peft_model(model, lora_config)
    
    return model


def finetune_model(model, tokenizer, train_dataset, eval_dataset=None):
    """Fine-tune the model using QLoRA"""
    
    # Data collator
    if MODEL_MODE == "text-generation":
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8
        )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(SAVE_MODEL_PATH, "checkpoints"),
        num_train_epochs=TRAIN_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=100 if eval_dataset else None,
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting fine-tuning...")
    trainer.train()
    
    return model
# ===== END ADDED =====


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
            # ===== MODIFIED: Fixed typo in line below =====
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
# ===== ADDED PARAMETERS FOR FINE-TUNING =====
DATASET_TRAIN_SPLIT = float(os.environ.get("DATASET_TRAIN_SPLIT", 0.8))  # Fraction for training
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", 3))
TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE", 2))
SAVE_MODEL_PATH = os.environ.get("SAVE_MODEL_PATH", "./FINETUNED_MODELS/" + DATASET_LANG + "/" + MODEL_SHORT_NAME + "/")
ENABLE_FINETUNING = os.environ.get("ENABLE_FINETUNING", "true").lower() == "true"
# ===== END ADDED =====

# PROGRAM
print("START PROGRAM...")
print("Logging into HuggingFace Hub...")
if HF_LOGIN_TOKEN:
    login(HF_LOGIN_TOKEN)
    print("Logged in!")
else:
    print("No HF_LOGIN_TOKEN provided, proceeding without login...")
os.makedirs(os.path.dirname(DESTINATION_PATH), exist_ok=True)
# ===== ADDED: Create save model directory =====
os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
# ===== END ADDED =====
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
# ===== ADDED: Load cleaned dataset =====
with open(CLEANED_PATH, 'r') as f:
    cleaned = list(json.load(f).items())
# ===== END ADDED =====

# defining inference
DATASET_END = len(ocred) if DATASET_END == -1 else int(DATASET_END)

# ===== ADDED: FINE-TUNING SECTION =====
if ENABLE_FINETUNING:
    print("\n===== STARTING FINE-TUNING PHASE =====")
    
    # Subset the data according to DATASET_START and DATASET_END
    ocred_subset = ocred[DATASET_START:DATASET_END]
    cleaned_subset = cleaned[DATASET_START:DATASET_END]
    
    # Create dictionary for easy lookup (key -> text mapping)
    cleaned_dict = {key: text for key, text in cleaned_subset}
    
    # Align datasets - only use samples that exist in both ocred and cleaned
    aligned_data = []
    for key, ocred_text in ocred_subset:
        if key in cleaned_dict:
            aligned_data.append((key, ocred_text, cleaned_dict[key]))
    
    print(f"Aligned {len(aligned_data)} samples from both datasets")
    
    # Split into train and validation
    train_size = int(len(aligned_data) * DATASET_TRAIN_SPLIT)
    
    # Shuffle for better training (with fixed seed for reproducibility)
    random.seed(42)
    random.shuffle(aligned_data)
    
    train_data = aligned_data[:train_size]
    val_data = aligned_data[train_size:] if DATASET_TRAIN_SPLIT < 1.0 else None
    
    print(f"Training samples: {len(train_data)}")
    if val_data:
        print(f"Validation samples: {len(val_data)}")
    
    # Extract only texts for dataset (ignore keys for training)
    train_ocred_texts = [item[1] for item in train_data]
    train_cleaned_texts = [item[2] for item in train_data]
    
    if val_data:
        val_ocred_texts = [item[1] for item in val_data]
        val_cleaned_texts = [item[2] for item in val_data]
    
    # Create datasets
    train_dataset = OCRCorrectionDataset(
        train_ocred_texts, 
        train_cleaned_texts, 
        tokenizer, 
        PROMPT
    )
    
    eval_dataset = None
    if val_data:
        eval_dataset = OCRCorrectionDataset(
            val_ocred_texts,
            val_cleaned_texts,
            tokenizer,
            PROMPT
        )
    
    # Setup QLoRA
    model = setup_qlora_config(model)
    print("QLoRA configuration applied")
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Fine-tune the model
    model = finetune_model(model, tokenizer, train_dataset, eval_dataset)
    
    # Save the fine-tuned adapter
    print(f"Saving fine-tuned model to {SAVE_MODEL_PATH}")
    model.save_pretrained(SAVE_MODEL_PATH)
    tokenizer.save_pretrained(SAVE_MODEL_PATH)
    
    print("===== FINE-TUNING COMPLETED =====\n")
    
    # For inference, we need to merge or use the adapter
    # The model is already loaded with adapter, ready for inference
else:
    print("Fine-tuning disabled, proceeding directly to inference...")
    
    # ===== ADDED: Check if a fine-tuned model exists and load it =====
    adapter_path = os.path.join(SAVE_MODEL_PATH, "adapter_config.json")
    if os.path.exists(adapter_path):
        print(f"Found existing fine-tuned model at {SAVE_MODEL_PATH}, loading it...")
        model = PeftModel.from_pretrained(model, SAVE_MODEL_PATH)
        print("Fine-tuned model loaded!")
    # ===== END ADDED =====
# ===== END FINE-TUNING SECTION =====

print(f"Running inference on {DATASET_END - DATASET_START} samples, from {DATASET_START} to {DATASET_END}, in batches of {BATCH_SIZE}...")
inferer = InferenceHandler(model, tokenizer, device=DEVICE)
results = inferer.run_batch(ocred[DATASET_START:DATASET_END], batch_size=BATCH_SIZE)
print(f"Generated {len(results)} results.")

# saving results
with open(destination_file, 'w+') as f:
    json.dump({key: result for key, result in results.items()}, f, indent=4)

print("Results saved to: " + destination_file)