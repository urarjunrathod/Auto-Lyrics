# <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Musical%20Notes.png" width="35px"> AutoLyrics

> **"Teaching machines to hear the words behind the melody."**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-8A2BE2?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/huggingface/peft)
[![Gradio](https://img.shields.io/badge/Gradio-Demo-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

---

[🚀 Features](#-features) • [🧬 How it Works](#-how-it-works) • [🛠 Setup](#-installation--setup) • [📁 Project Structure](#-project-structure) • [📊 Evaluation](#-evaluation) • [👥 Team](#-team)

---

## 🧭 The Problem

Standard Automatic Speech Recognition (ASR) systems are optimized for spoken language — clean, rhythmic, predictable. Singing is the opposite. It introduces:

- **Pitch variations** — vowels stretch across notes, breaking phoneme boundaries
- **Prolonged phonemes** — a single syllable can last multiple beats
- **Background instrumentation** — drums, bass, and harmonics compete with the voice
- **Rhythm and melody distortions** — stress patterns don't follow natural speech

The result? Off-the-shelf ASR models produce garbled, inaccurate lyric transcriptions that miss words, merge syllables, and fail on musical phrasing.

**AutoLyrics** solves this by fine-tuning an open-source ASR Transformer specifically on singing audio — using lightweight LoRA adapters so the model learns the nuances of sung speech without full retraining.

---

## 🚀 Features

- Singing voice transcription from short music clips
- LoRA-based parameter-efficient fine-tuning — trains in minutes, not hours
- HuggingFace Transformer integration for easy model swapping
- Full audio preprocessing and dataset handling pipeline
- Evaluation using WER (Word Error Rate) and CER (Character Error Rate)
- Interactive Gradio demo — drop in any audio clip and get lyrics instantly
- Support for custom audio inference on your own music

---

## 🧬 How it Works

```
🎵 Singing Audio Clip
        │
        ▼
🔧 Audio Preprocessing (Torchaudio)
        │  Resample, normalize, extract mel-spectrogram features
        ▼
🤖 ASR Transformer (HuggingFace)
        │  Whisper or similar pretrained ASR model
        ▼
⚡ LoRA Fine-Tuning (PEFT)
        │  Low-Rank Adapters trained on SLT Lyrics Dataset
        │  Only adapter weights updated — base model frozen
        ▼
📝 Lyric Transcription Output
        │
        ▼
📊 Evaluation (Jiwer)
        WER and CER compared against baseline pretrained model
```

---

## 🏗 Architecture

| Component | Technology | Role |
|---|---|---|
| **Audio Processing** | Torchaudio | Resampling, feature extraction, mel-spectrograms |
| **Base Model** | HuggingFace Transformers | Pretrained ASR Transformer backbone |
| **Fine-Tuning** | PEFT (LoRA) | Parameter-efficient adapter training |
| **Quantization** | BitsAndBytes | Memory-efficient model loading |
| **Dataset** | HuggingFace Datasets | SLT Lyrics Audio dataset loading and splits |
| **Evaluation** | Jiwer | WER and CER metric computation |
| **Demo** | Gradio | Interactive inference interface |

---

## 📁 Project Structure

```
autolyrics/
├── notebook/
│   └── autolyrics.ipynb        # Main training and evaluation notebook
│
├── data/
│   └── slt_lyrics/             # Downloaded dataset cache
│
├── models/
│   └── lora_adapters/          # Saved LoRA adapter weights after training
│
├── demo/
│   └── app.py                  # Gradio demo application
│
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📦 Dataset

The project uses the **SLT Lyrics Audio Dataset** available on HuggingFace:

🔗 https://huggingface.co/datasets/gmenon/slt-lyrics-audio

The dataset contains:
- Singing audio clips paired with lyric transcriptions
- Audio-text pairs suitable for ASR fine-tuning
- Pre-split into training, validation, and test sets

Used for training the LoRA adapters, validating during fine-tuning, and evaluating WER/CER against the pretrained baseline.

---

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/yourusername/autolyrics.git
cd autolyrics
```

---

### Step 2 — Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

---

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install torch torchaudio transformers datasets peft bitsandbytes jiwer gradio
```

---

### Step 4 — Run the Notebook

Open and run the training notebook:
```bash
jupyter notebook notebook/autolyrics.ipynb
```

The notebook walks through:
1. Dataset loading and preprocessing
2. Audio feature extraction
3. Model loading with LoRA configuration
4. Fine-tuning pipeline
5. Evaluation against baseline
6. Custom inference testing

---

### Step 5 — Launch the Gradio Demo

```bash
python demo/app.py
```

Opens an interactive interface at `http://localhost:7860` — upload any singing audio clip and get the transcribed lyrics instantly.

---

## 📊 Evaluation

Model performance is measured using two standard ASR metrics:

| Metric | Description | Lower is Better |
|---|---|---|
| **WER** (Word Error Rate) | % of words incorrectly transcribed | ✅ |
| **CER** (Character Error Rate) | % of characters incorrectly transcribed | ✅ |

Results are compared between:
- **Baseline** — pretrained ASR model with no fine-tuning
- **AutoLyrics** — same model with LoRA adapters trained on singing data

---

## ⚡ Why LoRA?

Full fine-tuning of a large ASR Transformer requires updating hundreds of millions of parameters — expensive in time, memory, and compute.

LoRA (Low-Rank Adaptation) inserts small trainable adapter matrices into the model's attention layers. Only these adapters are updated during training — the base model stays frozen. This means:

- Training is **10-100x faster** than full fine-tuning
- **GPU memory requirements drop significantly**
- The original model weights are preserved — adapters can be swapped
- Adapter files are tiny (a few MB vs gigabytes for full model weights)

---

## 🔮 Future Improvements

- Support for full song transcription (not just short clips)
- Multi-language lyric transcription
- Speaker diarization to separate lead vocals from backing vocals
- Real-time streaming transcription
- Dataset expansion with more diverse music genres
- Timestamp alignment — syncing transcribed words to audio position

---

## 👥 Team

Built as part of a college project submission.

| Role | Contribution |
|---|---|
| ML Engineer | LoRA fine-tuning, audio preprocessing, evaluation pipeline |
| ML Engineer | Dataset handling, model configuration, Gradio demo |

---

## 📄 License

This project is licensed under the MIT License.

---

*"Every song has words. Now every model can hear them."*
**— The AutoLyrics Team**
