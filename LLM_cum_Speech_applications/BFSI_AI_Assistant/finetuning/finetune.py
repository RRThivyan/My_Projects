# =================================
# PHI-3.5-MINI-INSTRUCT FINETUNING
# =================================

import torch
import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
print("✅ Libraries loaded")

# Check versions
import transformers
print(f"Transformers version: {transformers.__version__}")

# ------------------------------------------------------------
# 1. Paths
# ------------------------------------------------------------
DATASET_PATH = "bfsi_cleaned_dataset.json"
SAVE_PATH = "phi35-bfsi-lora"
os.makedirs(SAVE_PATH, exist_ok=True)

# ------------------------------------------------------------
# 2. Load and Prepare Dataset
# ------------------------------------------------------------
print("📂 Loading dataset...")
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"📊 Loaded {len(data)} records")

def format_conversation(example):
    """Format Alpaca data into conversation format with special tokens"""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    # Create conversation with special tokens
    if input_text and input_text.strip():
        user_msg = f"<|user|>\n{instruction}\n\n{input_text}\n<|end|>\n"
    else:
        user_msg = f"<|user|>\n{instruction}\n<|end|>\n"

    assistant_msg = f"<|assistant|>\n{output}\n<|end|>"

    full_conversation = user_msg + assistant_msg

    return {"text": full_conversation}

# Format all examples
formatted_data = []
for item in data:
    try:
        formatted_data.append(format_conversation(item))
    except Exception as e:
        print(f"⚠️ Error formatting item: {e}")
        continue

print(f"✅ Formatted {len(formatted_data)} records")

# Create dataset
dataset = Dataset.from_list(formatted_data)

# Split dataset
dataset = dataset.train_test_split(test_size=0.05, seed=42)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print(f"📊 Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")

# Print sample
print("\n📄 Sample formatted conversation:")
print(train_dataset[0]["text"])
print("-" * 60)

# ------------------------------------------------------------
# 3. Load Model and Tokenizer (with quantization)
# ------------------------------------------------------------
print("🔄 Loading model and tokenizer...")
model_name = "microsoft/Phi-3.5-mini-instruct"

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=False,
    padding_side="right"
)

# Add special tokens
special_tokens = ["<|user|>", "<|assistant|>", "<|end|>"]
tokenizer.add_tokens(special_tokens)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# Resize embeddings for new tokens
model.resize_token_embeddings(len(tokenizer))
print(f"✅ Resized embeddings to {len(tokenizer)} tokens")

print("✅ Model loaded in 4-bit")

# ------------------------------------------------------------
# 4. Prepare model for k-bit training and add LoRA
# ------------------------------------------------------------
print("🔧 Configuring LoRA...")

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["embed_tokens", "lm_head"],  # Important for new tokens
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.config.use_cache = False

model.print_trainable_parameters()

# ------------------------------------------------------------
# 5. Tokenize Dataset
# ------------------------------------------------------------
print("📝 Tokenizing dataset...")

def tokenize_function(element):
    """Tokenize text with proper truncation"""
    return tokenizer(
        element["text"],
        truncation=True,
        max_length=1024,
        add_special_tokens=True,
    )

# Tokenize datasets
print("🔄 Tokenizing training dataset...")
tokenized_train_dataset = train_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=["text"],
)

print("🔄 Tokenizing evaluation dataset...")
tokenized_eval_dataset = eval_dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=["text"],
)

print("✅ Tokenization complete")

# ------------------------------------------------------------
# 6. Custom Collate Function
# ------------------------------------------------------------
def collate_fn(elements):
    """Custom collate function with dynamic padding"""
    input_ids_list = [e["input_ids"] for e in elements]
    attention_mask_list = [e["attention_mask"] for e in elements]

    # Find max length in batch
    max_len = max([len(ids) for ids in input_ids_list])

    # Pad sequences
    padded_input_ids = []
    padded_attention_masks = []
    labels_list = []

    for input_ids, attention_mask in zip(input_ids_list, attention_mask_list):
        pad_len = max_len - len(input_ids)

        # Pad input_ids with pad_token_id
        padded_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        padded_input_ids.append(padded_ids)

        # Pad attention_mask with 0
        padded_mask = attention_mask + [0] * pad_len
        padded_attention_masks.append(padded_mask)

        # For causal LM, labels are same as input_ids (with -100 for padding)
        labels = input_ids + [-100] * pad_len
        labels_list.append(labels)

    # Convert to tensors
    batch = {
        "input_ids": torch.tensor(padded_input_ids),
        "attention_mask": torch.tensor(padded_attention_masks),
        "labels": torch.tensor(labels_list),
    }

    return batch

# ------------------------------------------------------------
# 7. Training Arguments - FIXED FOR TRANSFORMERS 5.0.0
# ------------------------------------------------------------
print("⚙️ Configuring training arguments...")

args = TrainingArguments(
    output_dir=SAVE_PATH,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    # Use eval_strategy (new in v5.0.0)
    eval_strategy="no",  # or "steps", "epoch" - "no" disables evaluation during training
)

print("✅ Training arguments configured")

# ------------------------------------------------------------
# 8. Initialize Trainer - FIXED FOR TRANSFORMERS 5.0.0
# ------------------------------------------------------------
print("🚀 Setting up Trainer...")

# In Transformers 5.0.0, 'tokenizer' is renamed to 'processing_class'
trainer = Trainer(
    model=model,
    processing_class=tokenizer,  # ⚠️ KEY CHANGE: tokenizer -> processing_class
    data_collator=collate_fn,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    args=args,
)

print("✅ Trainer initialized")

# ------------------------------------------------------------
# 9. Train the model
# ------------------------------------------------------------
print("🚀 Starting Fine-Tuning...")
print("=" * 60)
print(f"Training on {len(tokenized_train_dataset)} samples")
print(f"Validation on {len(tokenized_eval_dataset)} samples")
print(f"Total trainable parameters: {model.num_parameters(only_trainable=True):,}")
print("=" * 60)

train_result = trainer.train()

# Log and save metrics
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# ------------------------------------------------------------
# 10. Final Evaluation
# ------------------------------------------------------------
print("\n📊 Running final evaluation...")
metrics = trainer.evaluate()
metrics["eval_samples"] = len(tokenized_eval_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

# ------------------------------------------------------------
# 11. Save model
# ------------------------------------------------------------
print("💾 Saving model...")
trainer.save_model(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

# Save the LoRA adapter separately
model.save_pretrained(os.path.join(SAVE_PATH, "lora_adapter"))
print(f"✅ Model saved to: {SAVE_PATH}")

# ------------------------------------------------------------
# 12. Test the model
# ------------------------------------------------------------
print("\n🧪 Testing the fine-tuned model...")

def test_model(prompt, model, tokenizer, max_new_tokens=200):
    """Test the model with a prompt"""
    # Format prompt with special tokens
    formatted_prompt = f"<|user|>\n{prompt}\n<|end|>\n<|assistant|>\n"

    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response

# Test prompts
test_prompts = [
    "What is the interest rate for a personal loan?",
    "How do I check my loan eligibility?",
    "Tell me about home loan EMI options",
    "What documents are needed for a home loan?",
]

print("\n" + "="*60)
print("TESTING THE MODEL")
print("="*60)

for i, prompt in enumerate(test_prompts):
    print(f"\n🔹 Test {i+1}: {prompt}")
    try:
        response = test_model(prompt, model, tokenizer)
        print(f"🔸 Assistant: {response}")
    except Exception as e:
        print(f"⚠️ Error generating response: {e}")
    print("-" * 60)

print("\n✅ Fine-tuning and testing complete!")