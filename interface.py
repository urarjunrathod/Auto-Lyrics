"""
Gradio inference interface for AutoLyrics
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

from transformers import pipeline

inference_model = model.to(device)
inference_model.eval()
inference_model.config.use_cache = True

baseline_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto" if device == "cuda" else None
)
baseline_model.eval()
baseline_model.config.use_cache = True

#Chunking
tuned_pipe = pipeline(
    "automatic-speech-recognition",
    model=inference_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    device=device
)

base_pipe = pipeline(
    "automatic-speech-recognition",
    model=baseline_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    device=device
)

def transcribe_long_audio(audio_file_path):
    if audio_file_path is None:
        return "No audio provided.", "No audio provided."


    tuned_result = tuned_pipe(
        audio_file_path,
        generate_kwargs={"max_new_tokens": 225, "num_beams": 3}
    )

    base_result = base_pipe(
        audio_file_path,
        generate_kwargs={"max_new_tokens": 225, "num_beams": 3}
    )

    return tuned_result["text"], base_result["text"]

#Launching Gradio interface
demo = gr.Interface(
    fn=transcribe_long_audio,
    inputs=gr.Audio(type="filepath", label="Upload Full Song"),
    outputs=[
        gr.Textbox(label="AutoLyrics Fine-Tuned Transcription"),
        gr.Textbox(label="Baseline (Un-tuned) Transcription")
    ],
    title="AutoLyrics: Long-Form Singing-Voice ASR"
)

demo.launch(share=True)
