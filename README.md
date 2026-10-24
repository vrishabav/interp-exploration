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

## Results so far

Model: Qwen2.5-1.5B-Instruct. Refusal is scored by substring matching (Arditi et al.'s list) on 64-token greedy
completions, and by a logit score (log-odds that the first token is "I" or "As"). The substring scorer misses some
soft refusals ("It is not appropriate..."), so treat refusal rates as approximate until a hand-labelled check is done.

**1. The refusal direction reproduces.** Difference of mean activations (harmful minus harmless, 128 train prompts each)
at every post-instruction position and layer, chosen with Arditi et al.'s filters (bypass, induce, KL < 0.1).
Only 1 of 105 candidates passed: the last post-instruction token, layer 17.

| on held-out prompts (100 each) | harmful refused | harmless refused | FineWeb loss |
|---|---|---|---|
| no intervention | 68% | 3% | 3.017 |
| direction ablated (all layers) | 3% | 3% | 3.060 |
| direction added (layer 17) | 99% | 77% | - |

**2. Attention and MLPs each write about half of the direction.** The harmful-minus-harmless gap along the
direction at layer 17 splits into attention writes (49%) and MLP neuron writes (51%). Each neuron's score is
`(mean h on harmful - mean h on harmless) x (its down_proj column . direction)`. The top 100 neurons carry 45% of
the total gap, and 72 of them sit in layers 15-16.

**3. A small set of neurons matters causally.** Mean-ablating the top k neurons (replacing h with its mean on
harmless prompts, at every token) vs. k random neurons from the same layers (3 seeds):

| k | top-k: harmful refused | random k: harmful refused |
|---|---|---|
| 5 | 63% | 69-70% |
| 25 | 43% | 69-70% |
| 100 | 15% | 67-70% |

FineWeb loss goes from 3.017 to 3.028 with the top 100 ablated. The effect is graded: there's no handful of
neurons that switches refusal off.

**4. The neurons' effect is mostly direct.** With the top 100 ablated, the measured drop in the gap at layer 17 is:

| measure | share of the full-direction effect |
|---|---|
| direct write predicted by attribution | 45% |
| measured gap drop, ablating only at the last token | 43% |
| measured gap drop, ablating at all tokens | 53% |
| logit refusal score (harmful val) | 60% |
| refusal rate (harmful test) | 82% |

So there are no hidden indirect paths at the last token, about 8 points come from earlier tokens, and the jump to
82% comes from refusal rate behaving like a threshold. Refusal rates overstate how much of the mechanism a set of
components carries.

**5. The negative pole carries most of the refusal signal.** Splitting each neuron's score by the signs of gate and
up (h = silu(gate) x up): for the top 100 neurons, 56% of the score comes from negative activations (almost all
gate > 0, up < 0) and under 2% from the double-negative case (gate < 0, up < 0). Looking only at a neuron's top
positive activations would miss more than half of what it does for refusal.

**6. Single neurons can't steer refusal.** Clamping any one of the top 20 neurons to its 99.9th-percentile value
(from 2M tokens of general text) on harmless prompts barely changes the refusal score. Only at 16x that value do a
few neurons push the model toward refusing.

**Still to do:** label and grade the top 20 neurons from `results/<model>/neuron_cards_*.md`, hand-check the refusal
scorer, and test the over-refusal (XSTest) and weak-vs-strong activation results, which are not solid yet.

## Data

- Harmful/harmless prompts: the pre-built splits from [andyrdt/refusal_direction](https://github.com/andyrdt/refusal_direction)
  (AdvBench, MaliciousInstruct, TDC, HarmBench, JailbreakBench / Alpaca).
- Over-refusal: [XSTest](https://github.com/paul-rottger/xstest).
- Audit corpus: `HuggingFaceFW/fineweb` (sample-10BT) and `HuggingFaceH4/ultrachat_200k`.

## AI Disclosure

AI Agents were used to:
- Assist with code implementation through this project
- Draft the README, which has been read through and edited by me
- Debug and troubleshoot

## References

- Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024). [arXiv:2406.11717](https://arxiv.org/abs/2406.11717)
- Cammarata (Goodfire), *Neural Networks Are Already Clean*. [already-clean.goodfire.com](https://already-clean.goodfire.com/)