# Results

Qwen2.5-1.5B-Instruct is used to study safety refusal and evidence-based abstention. Safety refusal
declines a harmful request. Evidence-based abstention recognizes that supplied records lack the
answer. The experiments ask whether training abstention increases dependence on neurons previously
identified as causal contributors to safety refusal.

## 1. Safety refusal

Harmful requests come from the prebuilt Arditi et al. splits; harmless instructions come from Alpaca.
Direction discovery uses 128 training prompts of each kind, selection uses 32 validation prompts of
each kind, and generation evaluation uses 100 test prompts of each kind.

The refusal direction is a harmful-minus-harmless mean difference in the residual stream, the vector
passed between transformer layers. One of 105 candidates passes the induction and KL filters:
layer 17 at the last prompt token. Its norm is 18.30 and harmless next-token KL is 0.075,
below the 0.1 threshold.

Refusal is detected by phrase matching on 64-token greedy responses. The separate logit score is
the log-odds of the first response token being “I” or “As”. FineWeb loss measures next-token
prediction on 64 general-web-text chunks of 128 tokens each; lower is better.

| Intervention | Harmful refusal, /100 | Harmless refusal, /100 | FineWeb loss |
|---|---:|---:|---:|
| None | 68 | 3 | 3.017 |
| Direction removed throughout the model | 3 | 3 | 3.060 |
| Direction added at layer 17 | 99 | 77 | — |

The direction strongly affects refusal. Phrase matching misses some soft refusals and can count
partial compliance as refusal, so these rates remain approximate.
[Direction validation](results/Qwen2.5-1.5B-Instruct/direction_validation_p4_L17.json)

### Neuron attribution and ablation

A neuron's score multiplies its harmful-minus-harmless activation difference by its output vector's
projection onto the refusal direction. MLP neurons contribute about 51% of the layer-17 gap and
attention about 49%. The top 100 write 45.2% of the total gap; 72 lie in layers 15–16.

Mean ablation replaces selected activations at every prompt and generated token with their
harmless-training means at the final prompt position. Three random controls match the selected
neurons' layer counts and exclude the top 100.

| Neurons ablated | Harmful refusal | Random refusal range | Harmful validation logit score | FineWeb loss |
|---|---:|---:|---:|---:|
| None | 68% | — | −0.55 | 3.017 |
| Top 5 | 63% | 69–70% | −1.28 | 3.017 |
| Top 10 | 54% | 69% | −1.44 | 3.017 |
| Top 25 | 43% | 69–70% | −1.96 | 3.017 |
| Top 50 | 34% | 69–71% | −2.31 | 3.025 |
| Top 100 | 15% | 67–70% | −2.99 | 3.028 |
| Full direction | 3% | — | −4.65 | 3.060 |

The effect grows with the number of selected neurons. Top-100 ablation increases the measured
language-model loss by about 0.38%. This capability check covers a small general-text sample.
[Causal results](results/Qwen2.5-1.5B-Instruct/causal_p4_L17.json)

### Direct writes and measured projection changes

These percentages use the original harmful–harmless projection gap on training prompts as their
denominator. “Final token” restricts the intervention to the last prompt position.

| Top neurons | Direct contribution predicted by attribution | Gap reduction, final token | Gap reduction, all positions |
|---|---:|---:|---:|
| 5 | 16.5% | 20.4% | 21.7% |
| 10 | 21.3% | 24.7% | 26.9% |
| 25 | 29.9% | 30.5% | 37.9% |
| 50 | 37.0% | 36.5% | 44.7% |
| 100 | 45.2% | 42.8% | 53.5% |

For the top 100, intervening at other positions adds 10.7 percentage points of net gap reduction.
The close attribution/final-token comparison allows compensating indirect pathways. Separately,
top-100 ablation achieves 81.5% of the full-direction intervention's refusal-rate reduction; that
behavioral ratio uses a different denominator and held-out prompts.
[Projection measurements](results/Qwen2.5-1.5B-Instruct/projection_under_ablation_p4_L17.csv)

Negative activations account for 55.7% of the top-100 attributed signal. The double-negative SwiGLU
case, where both gate and up inputs are negative, accounts for 2.1%. These are descriptive sign
decompositions. Human legibility labels for the top 20 neurons remain unfinished.
[Sign decomposition](results/Qwen2.5-1.5B-Instruct/quadrants_p4_L17.csv)

## 2. Evidence tasks and training

Each prompt contains five fictional entity records, plus five group records for two-lookup questions.
Training and pilot examples map trains to platforms or events to rooms. Tests map parcels to bins.
The reasoning structures are shared, so this measures transfer across records and vocabulary.

For example, “Parcel 4100 belongs to group 237” and “Group 237 uses bin A6” establish the answer A6.
Replacing the group-237 record with an unrelated group makes the bin undetermined, even though the
parcel is present. A separate failure type removes the parcel's record.

Each run has 256 training pairs, 32 pilot pairs and 128 test pairs. A pair asks the same question
with sufficient or insufficient records. Each test therefore has 256 prompts:

| Prompt type | Count | Required behavior |
|---|---:|---|
| Direct mapping present | 64 | Return the recorded bin. |
| Both links present | 64 | Follow parcel → group → bin. |
| Parcel absent, direct task | 64 | Abstain. |
| Parcel absent, two-lookup task | 32 | Abstain. |
| Parcel present, group-to-bin mapping absent | 32 | Abstain. |

The model must return a code or INSUFFICIENT. The verifier accepts the correct code with an optional
matching field name, normalizes case and a final period, and gives one point for correctness.
Other responses receive zero. Pair accuracy requires both members to be correct.

SFT (supervised fine-tuning) learns verified targets. RLOO (REINFORCE Leave-One-Out) samples four
responses per prompt and compares each reward with its siblings' mean. Both train rank-8 LoRA
adapters on attention and MLP projections, using four prompts per update and 128 updates.
SFT uses learning rate 0.0001; RLOO uses 0.00003 and sequence KL coefficient 0.02 against the original
instruct model. Training responses are capped at 16 tokens. Test decoding is greedy.
The first run uses seed 37; the follow-up uses seed 83 and fresh records.
[First configuration](results/rl_summary/baseline/manifest.json) · [Follow-up configuration](results/rl_summary/warmup/manifest.json)

## 3. First comparison

Frozen means the original checkpoint. SFT and RLOO start independently from it.

| Measure | Frozen | SFT | RLOO |
|---|---:|---:|---:|
| Correct answers, /128 | 106 | 128 | 121 |
| Correct abstentions, all unanswerable cases, /128 | 96 | 128 | 96 |
| Correct abstentions, missing group, /32 | 0 | 32 | 0 |
| Both pair members correct, /128 | 83 | 128 | 93 |

RLOO improves answer accuracy while the missing-group failure persists. On eight missing-group pilot
prompts, all four samples per prompt earn zero reward. Identical task rewards give those groups no
relative task-reward signal at that point.
[Aggregate results](results/rl_summary/baseline/metrics.csv) · [Notebook](notebooks/02_abstention_rl.ipynb)

## 4. Warm-up followed by RLOO

The warm-up pool contains eight matched pairs from separate rail/venue records. The earliest tested
warm-up, two updates of four examples each, reaches 25% sampled reward on eight missing-group pilot
prompts. Six of those eight groups contain both successful and unsuccessful responses. That checkpoint
becomes the warm-up control and the starting policy for RLOO.

| Measure | Frozen | Warm-up | Warm-up + RLOO |
|---|---:|---:|---:|
| Correct answers, /128 | 112 | 61 | 125 |
| Correct abstentions, all unanswerable cases, /128 | 97 | 112 | 96 |
| Both pair members correct, /128 | 88 | 55 | 96 |

| Prompt type | Frozen | Warm-up | Warm-up + RLOO |
|---|---:|---:|---:|
| Direct mapping present, /64 | 64 | 37 | 64 |
| Both links present, /64 | 48 | 24 | 61 |
| Parcel absent, direct task, /64 | 64 | 64 | 64 |
| Parcel absent, two-lookup task, /32 | 32 | 32 | 32 |
| Group-to-bin mapping absent, /32 | 1 | 16 | 0 |

RLOO recovers answer accuracy and loses the warm-up's missing-group abstention. On those 32 cases,
16 regress and none improve. The paired change is −50 percentage points, with a 95% bootstrap
interval of −68.8 to −34.4 points. Post-RL sampled reward on the missing-group pilot also falls to zero.
These measurements establish the tradeoff in this run; its cause remains unresolved.
[Subtype results](results/rl_summary/warmup/subtype_metrics.csv) · [Paired comparison](results/rl_summary/warmup/rl_vs_warmup.json)

## 5. Do the original safety neurons matter more after training?

Interventions reuse the original 100 neuron IDs and layer-17 direction. Each checkpoint supplies its
own harmless-prompt replacement means. “Excess effect” is accuracy under random ablation minus
accuracy under original-neuron ablation, averaged over three random controls. A positive increase
suggests greater dependence on the original set.

| Change in excess effect on all unanswerable prompts | Estimate | 95% paired bootstrap interval |
|---|---:|---:|
| First-run RLOO versus frozen | +0.26 pp | −1.04 to +2.08 pp |
| Follow-up RLOO versus its warm-up | +1.56 pp | −1.82 to +4.95 pp |

Both intervals include zero, giving no clear evidence of increased reliance on the original set.
The interventions do move the internal projection: in the follow-up, original-neuron ablation reduces
the unanswerable-minus-answerable layer-17 gap from 0.845 to 0.760 after RLOO, while overall abstention
accuracy remains 75%.
[First-run estimates](results/rl_summary/baseline/recruitment.csv) · [Warm-up comparison](results/rl_summary/warmup/recruitment_vs_warmup.json) · [Intervention metrics](results/rl_summary/warmup/metrics.csv)

## 6. Human review and limits

A separate instruction asks for brief ordinary-language responses, capped at 48 tokens. The reviewer
sees records and responses with model identities hidden. Labels distinguish supported answers,
justified or unnecessary abstentions, unsupported answers, and unclear responses.

Each condition below covers ten answerable and ten unanswerable prompts. The first review contains
100 outputs across five conditions. The second deduplicates those outputs into 25 distinct
prompt–answer combinations, then maps labels back to all 100 outputs.

| Review | Model / intervention | Supported answers, /10 | Justified abstentions, /10 |
|---|---|---:|---:|
| First | Frozen, intact or original neurons ablated | 7 | 8 |
| First | RLOO, intact, original or random neurons ablated | 9 | 8 |
| Follow-up | Warm-up, intact | 8 | 7 |
| Follow-up | Warm-up, original neurons ablated | 7 | 7 |
| Follow-up | Warm-up + RLOO, intact, original or random neurons ablated | 10 | 7 |

All labels were checked against the records. Both first-run models guess on the two reviewed
missing-group cases. Both follow-up models guess on all three such cases, limiting the demonstrated
wording transfer of the strict-format warm-up result.
[First review](results/rl_summary/baseline/human_review.csv) · [Follow-up review](results/rl_summary/warmup/human_review.csv)

Each experiment uses one training seed. Bootstrap intervals resample record pairs and do not measure
training-seed variability. The runs use different test records, so comparisons belong within each run.
The review covers ten pairs per run, with repeated conditions on each pair. Generalization to broader
tasks, the mechanism of the RL tradeoff, and formal validation of the original refusal scorer remain open.

Note: AI agents were used to summarise and refine the presentation of these results.