"""Pilot selection and per-case diagnostics for the supervised warm-up experiment."""
import torch
import pandas as pd
from peft import set_peft_model_state_dict
from src.abstention_training import adapter_state, atomic_write, rollout, save_json, train_arm


def subtype(row):
    if row["answerable"]:
        return "linked_answer" if row["linked"] else "direct_answer"
    if not row["linked"]:
        return "direct_missing"
    records, question = row["task"].lower().split("\n\n")
    entity = " ".join(question.rstrip("?").split()[-2:])
    present = any(line.startswith(entity + " |") for line in records.splitlines())
    return "missing_group" if present else "missing_entity"


def sample_pilot(model, tok, cases, cfg):
    torch.manual_seed(cfg["seed"] + 10000)
    records = []
    for row in cases:
        _, _, mask, rewards, grades = rollout(model, tok, row, cfg)
        records.append(dict(pair=row["pair"], subtype=subtype(row),
                            reward=rewards.mean().item(),
                            mixed=float(rewards.max() > rewards.min()),
                            invalid=sum(g["invalid"] for g in grades)/len(grades),
                            tokens=mask.sum().item()))
    return records


def select_warmup(model, tok, demos, pilot, cfg, folder, initial):
    """Earliest scheduled checkpoint with positive, imperfect missing-group reward."""
    if (folder / "selection.json").exists():
        import json
        selection = json.loads((folder / "selection.json").read_text())
        return selection
    for steps in cfg["warmup_steps"]:
        warm_cfg = dict(cfg, steps=steps, save_every=steps)
        log = train_arm("sft", model, tok, demos, warm_cfg, folder/f"warmup_{steps}.pt", initial)
        records = sample_pilot(model, tok, pilot, cfg)
        save_json(folder/f"pilot_warmup_{steps}.json", records)
        hard = [r for r in records if r["subtype"]=="missing_group"]
        reward = sum(r["reward"] for r in hard)/len(hard)
        mixed = sum(r["mixed"] for r in hard)/len(hard)
        print(f"Warm-up {steps} steps: missing-group reward {reward:.3f}, mixed groups {mixed:.3f}", flush=True)
        if 0 < reward < 0.95 and mixed > 0:
            state = dict(adapter=adapter_state(model), log=log)
            atomic_write(folder/"sft.pt", lambda tmp: torch.save(state, tmp))
            selection = dict(steps=steps, missing_group_reward=reward, mixed=mixed)
            save_json(folder/"selection.json", selection)
            return selection
    raise RuntimeError("Warm-up pilot produced no partial-success checkpoint. Review the saved pilot results before extending the schedule.")


def subtype_metrics(frame):
    frame = frame.copy()
    frame["subtype"] = [subtype(row) for row in frame.to_dict("records")]
    return frame.groupby(["arm","condition","subtype"]).agg(
        correct=("correct","sum"), count=("correct","size"),
        accuracy=("correct","mean"), invalid=("invalid","mean")).reset_index()
