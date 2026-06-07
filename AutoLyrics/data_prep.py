"""
Dataset preprocessing module for AutoLyrics
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

#Loading the dataset (Hugging Face)
raw_dataset = load_dataset(DATASET_NAME)

#Resampling
raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))

#Processor (Feature extraction and Text tokenization)
processor = WhisperProcessor.from_pretrained(MODEL_NAME, language="english", task="transcribe")

def prepare_dataset(batch):
    audio = batch["audio"]

    #Input feautures
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    #Tokenization
    target_text = batch.get("text", batch.get("lyrics", ""))
    batch["labels"] = processor.tokenizer(target_text).input_ids
    return batch

#Select a subset if the dataset is large, or use full splits
train_split = raw_dataset["train"].select(range(min(20, len(raw_dataset["train"]))))
test_split = raw_dataset["test"].select(range(min(10, len(raw_dataset["test"])))) if "test" in raw_dataset else train_split

train_dataset = train_split.map(prepare_dataset, remove_columns=train_split.column_names)
eval_dataset = test_split.map(prepare_dataset, remove_columns=test_split.column_names)

#Data collator to dynamically pad inputs and labels
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
