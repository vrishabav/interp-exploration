# Refusal neurons

Breaking the refusal direction (Arditi et al., 2024) down to individual SwiGLU MLP neurons in
Qwen2.5-1.5B-Instruct, and checking how clean those neurons are.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell GPUs need cu128+
pip install -r requirements.txt
python -m ipykernel install --user --name refusal-neurons
```

Then open `refusal_neurons.ipynb` with the `refusal-neurons` kernel and run all cells. On first run it downloads
the model (~3 GB), the prompt splits and XSTest, streams ~2M tokens of FineWeb/UltraChat, and caches activations.
Later runs reload all of these from disk.

## Layout

```
refusal_neurons.ipynb           everything: config, data, hooks, caches, exploration
data/raw/                       downloaded prompt splits + XSTest csv
data/prompts.json               sampled train/val/test prompts
data/corpus_tokens.pt           general-text token chunks for the neuron audit
cache/<model>/acts_<split>.pt   resid / h / gate / up at post-instruction positions
cache/<model>/means.pt          per-split means + candidate refusal directions
results/<model>/generations/    cached completions (git-ignored)
figures/
```

## Data

- Harmful/harmless prompts: the pre-built splits from [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
  (AdvBench, MaliciousInstruct, TDC, HarmBench, JailbreakBench / Alpaca).
- Over-refusal: [XSTest](https://github.com/paul-rottger/xstest).
- Audit corpus: `HuggingFaceFW/fineweb` (sample-10BT) and `HuggingFaceH4/ultrachat_200k`.

## References

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024). [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Cammarata (Goodfire), *Neural Networks Are Already Clean*. [already-clean.goodfire.com](https://already-clean.goodfire.com/)
