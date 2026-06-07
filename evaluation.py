"""
Evaluation module for AutoLyrics
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


def evaluate_predictions(base_model_name, adapter_path, eval_data):
    #Loading both models
    base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name).to(device)

    peft_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(peft_model, adapter_path).to(device)

    references = []
    base_preds = []
    lora_preds = []

    #Run comparative inference on test samples
    for sample in eval_data.select(range(min(5, len(eval_data)))):
        input_feats = torch.tensor([sample["input_features"]]).to(device)

        #Ground Truth Text
        ref_ids = [i for i in sample["labels"] if i != -100]
        ref_text = processor.tokenizer.decode(ref_ids, skip_special_tokens=True)
        references.append(ref_text)

        #Zero-Shot Base Inference
        with torch.no_grad():

            base_gen = base_model.generate(input_features=input_feats)
            base_pred = processor.tokenizer.decode(base_gen[0], skip_special_tokens=True)
            base_preds.append(base_pred)

            # LoRA Fine-Tuned Inference
            lora_gen = peft_model.generate(input_features=input_feats)
            lora_pred = processor.tokenizer.decode(lora_gen[0], skip_special_tokens=True)
            lora_preds.append(lora_pred)

    #Compute Error Rates
    base_wer = wer(references, base_preds)
    lora_wer = wer(references, lora_preds)
    base_cer = cer(references, base_preds)
    lora_cer = cer(references, lora_preds)

    print("Results:")
    print(f"Base Model ({base_model_name}):- WER: {base_wer:.4f} %| CER: {base_cer:.4f} %")
    print(f"LoRA Fine-Tuned Model:- WER: {lora_wer:.4f} %| CER: {lora_cer:.4f} %")
    if base_wer > 0:
        improvement = ((base_wer - lora_wer) / base_wer) * 100
        print(f"Relative WER Reduction: {improvement:.2f}%")


evaluate_predictions(MODEL_NAME, OUTPUT_DIR, eval_dataset)
