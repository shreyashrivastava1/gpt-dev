# GPT and Bigram Language Model from Scratch (TinyShakespeare)

This project implements two language models trained on the TinyShakespeare dataset:

- A **Bigram Language Model** using token-level embeddings.
- A **GPT-style Transformer Language Model** with multi-head self-attention and positional embeddings.

## 🚀 Overview

- Framework: **PyTorch**
- Dataset: **Tiny Shakespeare** (character-level)
- Trained a transformer from scratch with:
  - 4 transformer blocks
  - 6 self-attention heads per block
  - 384-dimensional embeddings
- Generated Shakespeare-like text from scratch after training

## 📁 Files

- `gpt_model.py`: Transformer GPT model
- `bigram_model.py`: Simple bigram model for baseline comparison
- `input.txt`: Dataset used for training (TinyShakespeare)
- `README.md`: You're here!

## 🧠 Architecture Highlights (GPT)

- Token and positional embeddings
- Multi-head self-attention mechanism with masking
- Feed-forward neural nets in each block
- LayerNorm and dropout regularization
- Text generation through greedy sampling

## 🧪 Training Details

| Model  | Training Loss | Val Loss | Max Tokens Generated |
| ------ | ------------- | -------- | -------------------- |
| Bigram | ~2.3          | ~2.4     | 500                  |
| GPT    | ~1.2          | ~1.3     | 500                  |

Training was done using the AdamW optimizer and character-level prediction via cross-entropy loss.

## 💬 Sample Output

```
ROMEO:
And I do the sun to seem his man,
When thou hast made, and so the smiling be,
To bear that nothing thou.

JULIET:
I’ll love thee gone, a noble friar...
```

## 🛠️ Setup

```bash
# Clone the repo
git clone https://github.com/your-username/gpt-from-scratch.git
cd gpt-from-scratch

# Install dependencies (only torch is required)
pip install torch
```

## 🧠 Learnings

- Built GPT-style attention layers from scratch
- Understood token embeddings, position embeddings, and masked self-attention
- Compared performance of basic vs. transformer-based models
