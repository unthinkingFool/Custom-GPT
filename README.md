<h1 align="center">GPT From Scratch</h1>

<p align="center">
  <b>A decoder-only Transformer built from first principles in pure PyTorch — and a notebook anyone can point at their own text.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/dependencies-2-success?style=flat-square" alt="Dependencies">
  <img src="https://img.shields.io/badge/license-MIT-black?style=flat-square" alt="License">
</p>

<p align="center">
  <i>No Hugging Face <code>AutoModel</code>. No <code>nn.Transformer</code>. Every block written by hand.</i>
</p>

---

## Why this repo exists

Most "build GPT" tutorials leave you with a model that works on **their** dataset and nothing else. Wiring it to your own text means untangling hardcoded paths, vocab sizes and shapes.

This repo is the opposite. Every knob lives in a single `config` dictionary at the top of the notebook. **Change one line — the path to your text file — and train a GPT on your own data.**

```python
config = {
    "dataset_path": "docs/tiny-shakespeare.txt",   # <-- your text goes here. That's it.
    ...
}
```

Your song lyrics, your chat exports, your thesis, your favourite novel. Drop in a `.txt`, run all cells, watch it start writing like you.

This is the follow-up to my from-scratch implementation of the original *Attention Is All You Need* encoder–decoder Transformer → **[AttentionIsAllYouNeed](https://github.com/unthinkingFool/AttentionIsAllYouNeed)**. Here I strip the encoder away and rebuild the architecture that actually powers GPT.

---

## The architecture

A GPT is the *Attention Is All You Need* decoder with the encoder — and therefore cross-attention — removed. What's left is beautifully simple:

```
         Input tokens  (batch, seq_len)
                 │
                 ▼
      ┌─────────────────────┐
      │  Token Embeddings   │   ids ──► 256-dim meaning vectors
      └─────────────────────┘
                 │  +
      ┌─────────────────────┐
      │ Positional Encoding │   sin/cos waves ──► "where am I?"
      └─────────────────────┘
                 │
                 ▼
  ╔══════════════════════════════════╗
  ║        Decoder Block  × N        ║
  ║                                  ║
  ║   ┌──────────────────────────┐   ║
  ║   │  Masked Self-Attention   │   ║   look backward only
  ║   │      (multi-head)        │   ║   never at the future
  ║   └──────────────────────────┘   ║
  ║          │ + residual            ║
  ║   ┌──────────────────────────┐   ║
  ║   │      Feed Forward        │   ║   256 → 1024 → 256
  ║   └──────────────────────────┘   ║
  ║          │ + residual            ║
  ╚══════════════════════════════════╝
                 │
                 ▼
      ┌─────────────────────┐
      │  Projection Layer   │   ──► one score per vocabulary word
      └─────────────────────┘
                 │
                 ▼
        Next-token probabilities
```

Every component is implemented from scratch in this repo:

| Component | What it does |
|---|---|
| `InputEmbeddings` | Maps token ids to learnable `d_model` vectors, scaled by `√d_model` |
| `PositionalEncoding` | Fixed sinusoidal signal so attention knows word order |
| `MultiHeadAttentionBlock` | Q/K/V projections, scaled dot-product attention, causal masking, head splitting and merging |
| `FeedForwardBlock` | Position-wise two-layer network with ReLU |
| `LayerNormalization` | Hand-written layer norm with learnable gain and bias |
| `ResidualConnection` | Pre-norm residual wrapper — the reason deep stacks train at all |
| `DecoderBlock` | Self-attention → feed forward, two residual connections |
| `Decoder` | The stack, plus final normalization |
| `ProjectionLayer` | `d_model` → vocabulary logits |
| `BuildGPT` | Assembles the model and applies Xavier initialization |

---

## Trained on tiny-shakespeare

The repo ships trained on `tiny-shakespeare.txt` — the classic ~1MB corpus of Shakespeare's collected plays. It's small enough to train on a laptop CPU and large enough to produce recognisably Shakespearean output.

```
Using device: cuda
Loaded 1115394 characters from tiny-shakespeare.txt
Roughly 202651 words
Loaded tokenizer from tokenizers\gpt_tokenizer.json
Total tokens in dataset: 261973
Training windows: 235711, validation windows: 26134
Model has 10,008,623 parameters
Epoch 1/20 - train loss: 3.8193 - val loss: 8.0356 - val perplexity: 3088.9
Epoch 2/20 - train loss: 2.1805 - val loss: 8.3352 - val perplexity: 4168.0
Epoch 3/20 - train loss: 1.9508 - val loss: 8.2677 - val perplexity: 3895.9
Epoch 4/20 - train loss: 1.8612 - val loss: 8.2202 - val perplexity: 3715.4
Epoch 5/20 - train loss: 1.8102 - val loss: 8.1819 - val perplexity: 3575.7
```

**Sample generation** (prompt: `"KING RICHARD"`, temperature `0.8`, top-k `20`):

```
KING RICHARD II : Thanks , gentle Somerset ; sweet Oxford , thanks . PRINCE EDWARD : And take his thanks that yet hath nothing else . Messenger : Prepare you , lords , for Edward is at hand . Ready to fight ; therefore be resolute . OXFORD : I thought
```

Untrained, the model produces uniform noise. After twenty epochs it produces character names, line breaks, iambic cadence and dialogue structure — all learned purely from predicting the next word.

---

## Quickstart

```bash
git clone https://github.com/unthinkingFool/gpt-from-scratch.git
cd gpt-from-scratch

pip install torch tokenizers

jupyter notebook gpt_train.ipynb
```

Run all cells. The notebook trains the model, saves it to `weights/`, and generates text.

> **Two dependencies.** `torch` for tensors and autograd, `tokenizers` for the vocabulary. Nothing else — no `transformers`, no `datasets`, no training framework hiding the details.

---

## Train it on your own data

Three steps, about thirty seconds of work.

**1.** Put a plain text file in `docs/`. Anything works — one long document, or one sample per line.

**2.** Point the config at it:

```python
config["dataset_path"] = "docs/my_data.txt"
```

**3.** Delete `tokenizers/gpt_tokenizer.json` so a fresh vocabulary is built for your text, then run all cells.

That's the whole process. The vocabulary size, the model's output layer and the train/validation split all adapt to your file automatically.

<details>
<summary><b>Working from a CSV instead?</b></summary>

```python
import pandas as pd

df = pd.read_csv("docs/my_data.csv")
text = "\n".join(df["text_column"].dropna().astype(str))

with open("docs/my_data.txt", "w", encoding="utf-8") as f:
    f.write(text)
```

Then set `config["dataset_path"] = "docs/my_data.txt"` and carry on.
</details>

---

## Configuration

Everything is tunable from one dictionary.

| Key | Default | What it controls |
|---|---|---|
| `dataset_path` | `docs/tiny-shakespeare.txt` | **The only thing you need to change** |
| `seq_len` | `64` | How many tokens of context the model can see |
| `d_model` | `256` | Width of every vector flowing through the model |
| `num_layers` | `4` | How many decoder blocks are stacked |
| `num_heads` | `4` | Parallel attention heads (must divide `d_model`) |
| `d_ff` | `1024` | Hidden width of the feed-forward block |
| `dropout` | `0.1` | Regularization strength |
| `batch_size` | `32` | Windows per training step |
| `num_epochs` | `20` | Passes over the dataset |
| `lr` | `3e-4` | Adam learning rate |

**Sizing guidance:**

- **Small dataset (< 1 MB):** keep the defaults. A bigger model will simply memorize the text.
- **Larger dataset (> 10 MB):** try `d_model=512`, `num_layers=6`, `seq_len=128`, `batch_size=16`.
- **Out of memory?** Halve `batch_size` first, then `seq_len`.

---

## How training actually works

The single idea behind GPT: **predict the next token.** The implementation turns that into something remarkably efficient.

Your text becomes one long stream of token ids. Every training example is a window of that stream, paired with the same window shifted one position left:

```
stream:   the   sun   rose   over   the   city
input :  [the   sun   rose   over]
label :  [sun   rose   over   the ]
```

Read column by column and each window becomes many training questions at once:

| Given | Predict |
|---|---|
| `the` | `sun` |
| `the sun` | `rose` |
| `the sun rose` | `over` |
| `the sun rose over` | `the` |

One window of 64 tokens is **64 supervised examples learned in a single forward pass** — which is precisely why the causal mask matters. It guarantees position *i* can never see position *i+1*, so the model can't read the answer it's being asked to predict. Remove the mask and training loss collapses to near zero while generated text turns to noise.

---

## Generation

Text is produced one token at a time: predict, append, feed back in, repeat.

```python
generate(model, tokenizer, "KING RICHARD", config, device,
         max_new_tokens=50, temperature=0.8, top_k=20)
```

| Setting | Effect |
|---|---|
| `temperature=0` | Greedy — always the highest-scoring token. Deterministic, often repetitive |
| `temperature=0.7–0.9` | The sweet spot — coherent but not robotic |
| `temperature=1.3+` | Adventurous, increasingly incoherent |
| `top_k=20` | Samples only from the 20 most likely tokens, filtering out nonsense |

A sliding window keeps the last `seq_len` tokens in view, so generation can run well past the context length.

---

## GPT vs. the original Transformer

Coming from my [encoder–decoder implementation](https://github.com/unthinkingFool/AttentionIsAllYouNeed), here's exactly what changes:

| | Attention Is All You Need | GPT (this repo) |
|---|---|---|
| Architecture | Encoder + Decoder | **Decoder only** |
| Attention per block | Self + **cross** + feed forward | Self + feed forward |
| Tokenizers | Two (source, target) | **One** |
| Training signal | Sentence pairs | Next token in a stream |
| Masks | Padding + causal | **Causal only** |
| Task | Translate a sentence | Continue a sequence |
| Inference | Encode once, decode stepwise | Decode stepwise |

Dropping cross-attention is the entire structural difference. Everything else — embeddings, positional encoding, multi-head attention, residuals, layer norm — carries over unchanged. That's the insight worth internalizing: GPT isn't a new architecture, it's the same one with a piece removed and an objective that scales.

---

## Project structure

```
gpt-from-scratch/
├── gpt_train.ipynb              # The entire project — model, training, generation
├── docs/
│   └── tiny-shakespeare.txt     # Training corpus (swap this for your own)
├── tokenizers/
│   └── gpt_tokenizer.json       # Built automatically on first run
├── weights/
│   └── gpt_model_final.pt       # Saved checkpoint
└── README.md
```

The notebook is fully self-contained and heavily commented — every block carries a plain-English explanation of what it does and why it's there. It's written to be read top to bottom as a lesson, not just executed.

---

## Roadmap

- [ ] **BPE tokenizer** — replaces `[UNK]` on unseen words and shrinks the vocabulary substantially
- [ ] **Weight tying** — share the embedding matrix with the output projection
- [ ] **Learning-rate warmup** with the schedule from the original paper
- [ ] **Gradient clipping** for stability on larger corpora
- [ ] **KV caching** to cut generation from O(n²) to O(n) per token
- [ ] **Attention visualization** — plot the head-by-head attention maps
- [ ] **Training curve plots** in the notebook

---

## Acknowledgements

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., *Improving Language Understanding by Generative Pre-Training* (2018)
- The `tiny-shakespeare` corpus, popularised by Andrej Karpathy

---

## Author

**Swapnil Das** — CSE undergraduate, BUET

<p>
  <a href="https://github.com/unthinkingFool"><img src="https://img.shields.io/badge/GitHub-unthinkingFool-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://www.linkedin.com/in/swapnil-das-603824236"><img src="https://img.shields.io/badge/LinkedIn-Swapnil%20Das-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
</p>

**Related work:** [Attention Is All You Need — from scratch](https://github.com/unthinkingFool/AttentionIsAllYouNeed) · an English→Bangla neural machine translation Transformer, implemented block by block.

---

<p align="center">
  <sub>If this helped you understand how GPT actually works, a ⭐ goes a long way.</sub>
</p>
