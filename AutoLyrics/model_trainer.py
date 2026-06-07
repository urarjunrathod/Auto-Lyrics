"""
LoRA fine-tuning module for AutoLyrics
"""

import os
import torch
import torchaudio
import gradio as gr
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from jiwer import wer, cer
from dataclasses import dataclass
from typing import Any, Dict, List, Union

MODEL_NAME = "openai/whisper-small"  #Base model
DATASET_NAME = "gmenon/slt-lyrics-audio"
OUTPUT_DIR = "./whisper-lora-autolyrics"
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
MAX_STEPS = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

from transformers import BitsAndBytesConfig


#Quantization configuration
quantization_config = None
if device == "cuda":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

#Model loading using the new quantization_config parameter
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto" if device == "cuda" else None
)

#Applying PEFT(LoRA)
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#Base Model
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto" if device == "cuda" else None
)

#Disable automatic generation config caching during training
model.config.use_cache = False
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

#Enabling the input gradients for checkpointing with LoRA
model.enable_input_require_grads()

#Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
)


#Wrapping the base model with PEFT/LoRA wrapper
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


#Evaluation metrics
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_score = 100 * wer(label_str, pred_str)
    return {"wer": wer_score}

#Forcefully strip ANY hidden dataset columns to prevent kwargs TypeErrors
train_dataset = train_dataset.select_columns(["input_features", "labels"])
eval_dataset = eval_dataset.select_columns(["input_features", "labels"])

#Configuring training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_steps=10,
    max_steps=MAX_STEPS,
    gradient_checkpointing=True,
    fp16=True if device == "cuda" else False,
    eval_strategy="steps",
    per_device_eval_batch_size=BATCH_SIZE,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    label_names=["labels"],
    remove_unused_columns=False,
)

#Initializing the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

#Training
trainer.train()

#saving the trained LoRA adapter weights
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print(f"Model and processor successfully saved to {OUTPUT_DIR}")
