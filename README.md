# 🦙 Fine-Tuning Llama 3.2 3B for Arabic Legal QA with Unsloth and LoRA

![GitHub](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)
![Unsloth](https://img.shields.io/badge/Unsloth-2x%20Faster%20Fine--Tuning-brightgreen)

Welcome to the repository for fine-tuning the **Llama 3.2 3B** model on an **Arabic Legal Question-Answering** dataset using **Unsloth** and **LoRA**! This project demonstrates how to efficiently adapt large language models for specialized tasks while minimizing computational resources.

---

## 🚀 **Project Overview**

This project focuses on fine-tuning the **Llama 3.2 3B** model to answer legal questions in Arabic. By leveraging **LoRA (Low-Rank Adaptation)** and **Unsloth**, we achieve **2x faster fine-tuning** with significantly reduced memory usage. The fine-tuned model is optimized for legal domain tasks and can generate accurate, context-aware responses in Arabic.

### Key Features:
- **LoRA Fine-Tuning**: Adapts only 1-10% of the model's parameters, reducing memory and computational costs.
- **Unsloth Optimization**: Speeds up training by 2x using advanced optimizations for attention mechanisms and matrix operations.
- **Arabic Legal Dataset**: Fine-tuned on a curated dataset of legal questions and answers in Arabic.
- **GGUF Export**: The model can be exported to GGUF format for efficient CPU inference using `llama.cpp`.

---

## 🛠️ **Technologies Used**

- **Model**: [Llama 3.2 3B](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit)
- **Fine-Tuning Library**: [Unsloth](https://github.com/unslothai/unsloth)
- **LoRA**: Low-Rank Adaptation for parameter-efficient fine-tuning.
- **Dataset**: [Arabic Legal QA Dataset](https://huggingface.co/datasets/AhmedBou/MMMLU_arabic_law_Instruct)
- **Training Framework**: Hugging Face `SFTTrainer` and `transformers`.
- **Quantization**: 4-bit quantization for reduced memory usage.
- **Inference**: GGUF format for CPU-based inference with `llama.cpp`.

---

## 📂 **Repository Structure**
.
├── AI_agent.ipynb # Notebook for inference and model interaction
├── Llama_3_2_3B_law_finetuning_with_Unsloth.ipynb # Main fine-tuning notebook
├── llama_3_2_dataset_creation.ipynb # Notebook for dataset preparation
├── README.md # This file
└── .idea/ # IDE configuration files (e.g., PyCharm)

---

## 🧠 **Methodology**

### 1. **LoRA (Low-Rank Adaptation)**
LoRA is a parameter-efficient fine-tuning technique that adds small, trainable matrices to specific layers of the model. This allows us to fine-tune only a fraction of the model's parameters, reducing memory usage and training time.

- **Target Layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- **Rank (`r`)**: 16
- **Alpha (`lora_alpha`)**: 16
- **Dropout**: 0 (disabled for optimal performance)

### 2. **Unsloth Optimization**
Unsloth accelerates fine-tuning by optimizing key operations such as attention mechanisms and matrix multiplications. It supports 4-bit quantization, gradient checkpointing, and mixed-precision training.

- **4-bit Quantization**: Reduces memory usage by loading the model in 4-bit precision.
- **Gradient Checkpointing**: Minimizes memory usage during training by recomputing gradients on-the-fly.
- **Mixed Precision**: Uses FP16 or BF16 for faster computations.

### 3. **Dataset Preparation**
The dataset consists of legal questions and answers in Arabic. Each example is formatted using a prompt template:

```plaintext
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Answer the following question in Arabic:

### Input:
{question}

### Response:
{answer}

4. Training
The model is fine-tuned using Hugging Face's SFTTrainer with the following parameters:

Batch Size: 2 (per device)

Gradient Accumulation Steps: 4

Learning Rate: 2e-4

Max Steps: 410 (~2 epochs)

Optimizer: AdamW 8-bit

5. Inference
After fine-tuning, the model can generate responses to legal questions in Arabic. A TextStreamer is used for real-time text generation.

6. GGUF Export
The fine-tuned model is exported to GGUF format for efficient CPU inference using llama.cpp. The quantization method used is q4_k_m.

🏁 Getting Started
Prerequisites
Python 3.8+

PyTorch 2.0+

Hugging Face transformers and datasets

Unsloth (pip install unsloth)

Installation
Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

Install dependencies:
pip install -r requirements.txt

Fine-Tuning
Run the Llama_3_2_3B_law_finetuning_with_Unsloth.ipynb notebook to fine-tune the model.

Inference
Use the AI_agent.ipynb notebook to interact with the fine-tuned model.

📊 Results
Training Time: ~30 minutes on a single GPU (Tesla T4)
Memory Usage: ~8.5 GB VRAM
Model Performance: The fine-tuned model achieves high accuracy in generating legal responses in Arabic.

🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

