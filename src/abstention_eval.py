"""Checkpoint-specific interventions, paired estimates, and a blinded review sheet."""
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from src.abstention_env import grade, instruction
from src.abstention_training import atomic_write, encode, generate, save_json
from src.refusal_common import hooks


@torch.no_grad()
def calibration(model, tok, prompts, layer_ids, batch):
    """Harmless activation means at the final prompt token of this checkpoint."""
    layers = model.get_base_model().model.layers
    sums = {}
    def capture(layer):
        def record(module, args, kwargs):
            values = args[0][:, -1].float().sum(0)
            sums[layer] = sums.get(layer, 0) + values
        return record
    with hooks([(layers[l].mlp.down_proj, capture(l)) for l in layer_ids]):
        for i in range(0, len(prompts), batch):
            enc = encode(tok, prompts[i:i+batch], model.device)
            pos = (enc.attention_mask.cumsum(-1) - 1).clamp_min(0)
            model.get_base_model().model(**enc, position_ids=pos, use_cache=False)
    return {l: v / len(prompts) for l, v in sums.items()}


def intervention(model, neurons, means):
    layers = model.get_base_model().model.layers
    groups = defaultdict(list)
    for l, i in neurons:
        groups[l].append(i)

    def replace(indices, values):
        def hook(module, args, kwargs):
            h = args[0].clone()
            h[..., indices] = values.to(h)
            return (h, *args[1:]), kwargs
        return hook
    return [(layers[l].mlp.down_proj, replace(ix, means[l][ix])) for l, ix in groups.items()]


@torch.no_grad()
def evaluate(model, tok, cases, pre, direction, site, batch, limit, natural=False):
    results = []
    layers = model.get_base_model().model.layers
    direction = direction.to(model.device)
    for i in range(0, len(cases), batch):
        rows = cases[i:i+batch]
        projections = []
        def probe(module, args, kwargs):
            # The first call is the prompt prefill. The others describe generated tokens.
            if not projections:
                x = args[0] if args else kwargs["hidden_states"]
                projections.extend((x[:, -1].float() @ direction).tolist())
        with hooks([*pre, (layers[site], probe)]):
            _, _, _, answers = generate(model, tok, [instruction(r, natural) for r in rows], limit)
        for row, answer, proj in zip(rows, answers, projections, strict=True):
            out = dict(**row, answer=answer, projection=proj)
            if not natural:
                out.update(grade(answer, row))
            results.append(out)
    return results


def summarize(df):
    out = []
    for (arm, condition), rows in df.groupby(["arm", "condition"], sort=False):
        a, u = rows[rows.answerable], rows[~rows.answerable]
        out.append(dict(arm=arm, condition=condition, answer_accuracy=a.correct.mean(),
                        correct_abstention=u.correct.mean(), invalid=rows.invalid.mean(),
                        pair_accuracy=rows.groupby("pair").correct.min().mean(),
                        projection_gap=u.projection.mean()-a.projection.mean()))
    return pd.DataFrame(out)


def recruitment(df, seed):
    """Matched pair IDs shared by arms/conditions; bootstrap across record sets."""
    out = []
    rng = np.random.default_rng(seed)
    for answerable, rows in df.groupby("answerable"):
        p = rows.pivot(index="pair", columns=["arm", "condition"], values="correct")
        controls = [c for c in p["frozen"] if c.startswith("random")]
        def excess(arm):
            return p[arm][controls].mean(axis=1) - p[arm]["original"]
        for arm in ["sft", "rloo"]:
            change = (excess(arm) - excess("frozen")).to_numpy()
            boot = rng.choice(change, (2000, len(change)), replace=True).mean(1)
            lo, hi = np.quantile(boot, [.025, .975])
            out.append(dict(arm=arm, answerable=answerable, change_pp=100*change.mean(),
                            low_pp=100*lo, high_pp=100*hi))
    return pd.DataFrame(out)


def review_sheet(natural_rows, folder, seed):
    """Preserve annotations on rerun; key and reviewer sheet are separate files."""
    df = pd.DataFrame(natural_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["review_id"] = [f"R{i+1:03}" for i in range(len(df))]
    key = df[["review_id","pair","arm","condition","answerable","target"]]
    sheet = df[["review_id","task","answer"]].copy()
    sheet["label"] = ""
    sheet["notes"] = ""
    path = folder / "review.csv"
    keypath = folder / "review_key.csv"
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        if not old[sheet.columns[:3]].equals(sheet.iloc[:, :3]):
            raise ValueError("Existing review sheet has different items. Use a separate run folder.")
    if keypath.exists() and keypath.read_text() != key.to_csv(index=False):
        raise ValueError("Existing review key has a different assignment.")
    atomic_write(keypath, lambda tmp: key.to_csv(tmp, index=False))
    if not path.exists():
        atomic_write(path, lambda tmp: sheet.to_csv(tmp, index=False))
    return path


def cached_evaluation(path, fn):
    if path.exists():
        return json.loads(path.read_text())
    rows = fn()
    save_json(path, rows)
    return rows
