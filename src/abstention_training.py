"""Single-GPU LoRA SFT and on-policy sequence-level RLOO."""
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import GenerationConfig

from src.abstention_env import grade, instruction
from src.refusal_common import format_instruction


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, data):
    atomic_write(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2)))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def adapter_state(model):
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def encode(tok, texts, device):
    return tok([format_instruction(x) for x in texts], add_special_tokens=False,
               return_tensors="pt", padding=True).to(device)


@torch.no_grad()
def generate(model, tok, texts, limit, group=1, sample=False):
    enc = encode(tok, texts, model.device)
    # Build a fresh config: inherited Qwen top-k, nucleus sampling and repetition
    # penalty would make rollout probabilities differ from the gradient calculation.
    eos = model.generation_config.eos_token_id
    config = GenerationConfig(do_sample=sample, max_new_tokens=limit, num_return_sequences=group,
                              pad_token_id=tok.pad_token_id, eos_token_id=eos,
                              top_k=0 if sample else None, repetition_penalty=1.0)
    start = enc.input_ids.shape[1]
    ids = model.generate(**enc, generation_config=config)[:, start:]
    stops = torch.isin(ids, torch.as_tensor(eos, device=ids.device).flatten())
    # Include the first EOS in the policy loss and exclude all subsequent padding.
    mask = (stops.long().cumsum(1) - stops.long()) == 0
    return enc, ids, mask, tok.batch_decode(ids, skip_special_tokens=True)


def response_logps(model, prompt_ids, ids, mask):
    # Training callers pass a single unpadded prompt. Only response logits are needed.
    full = torch.cat((prompt_ids.expand(len(ids), -1), ids), dim=1)
    logits = model(full, logits_to_keep=ids.shape[1] + 1, use_cache=False).logits[:, :-1].float()
    selected = logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return ((selected - logits.logsumexp(-1)) * mask).sum(-1)


def advantages(rewards):
    if len(rewards) < 2:
        raise ValueError("RLOO requires at least two samples per prompt")
    return rewards - (rewards.sum() - rewards) / (len(rewards) - 1)


def rollout(model, tok, row, cfg):
    enc, ids, mask, answers = generate(model, tok, [instruction(row)], cfg["tokens"], cfg["group"], True)
    grades = [grade(a, row) for a in answers]
    rewards = torch.tensor([g["correct"] for g in grades], device=ids.device, dtype=torch.float32)
    return enc.input_ids, ids, mask, rewards, grades


def rloo_gradient(model, tok, row, cfg, batch_size):
    prompt, ids, mask, rewards, grades = rollout(model, tok, row, cfg)
    with torch.no_grad():
        policy = response_logps(model, prompt, ids, mask)
        with model.disable_adapter():
            reference = response_logps(model, prompt, ids, mask)
        # Sampled sequence log ratio estimates KL on-policy. Individual values may be negative.
        log_ratio = policy - reference
        adv = advantages(rewards - cfg["beta"] * log_ratio)
    for j in range(len(ids)):
        lp = response_logps(model, prompt, ids[j:j+1], mask[j:j+1])
        (-adv[j] * lp / (batch_size * len(ids))).backward()
    return dict(reward=rewards.mean().item(), mixed=float(rewards.max() > rewards.min()),
                kl_sequence=log_ratio.mean().item(), invalid=sum(g["invalid"] for g in grades) / len(ids),
                response_tokens=mask.sum().item())


def sft_gradient(model, tok, row, cfg, batch_size):
    prompt = encode(tok, [instruction(row)], model.device).input_ids
    tokens = tok.encode(row["target"], add_special_tokens=False) + [tok.convert_tokens_to_ids("<|im_end|>")]
    ids = torch.tensor([tokens], device=model.device)
    lp = response_logps(model, prompt, ids, torch.ones_like(ids))
    (-lp / (batch_size * len(tokens))).backward()
    return dict(nll_per_token=-lp.detach().item() / len(tokens), response_tokens=len(tokens))


def train_arm(arm, model, tok, cases, cfg, path, initial):
    """One update per fresh batch. Resume at the last atomically saved optimizer step."""
    if arm not in {"sft", "rloo"}:
        raise ValueError(arm)
    # Eval mode keeps dropout disabled; gradients are enabled explicitly.
    model.eval()
    set_peft_model_state_dict(model, initial)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg[f"{arm}_lr"], weight_decay=0)
    log = []
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        set_peft_model_state_dict(model, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        log = state["log"]
        del state
    indices = list(range(len(cases)))
    random.Random(cfg["seed"]).shuffle(indices)
    batch = cfg["prompts_per_step"]
    for step in range(len(log), cfg["steps"]):
        torch.manual_seed(cfg["seed"] + step)
        start = time.perf_counter()
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        optimizer.zero_grad(set_to_none=True)
        grad = sft_gradient if arm == "sft" else rloo_gradient
        with torch.enable_grad():
            stats = [grad(model, tok, cases[indices[(step * batch + j) % len(indices)]], cfg, batch)
                     for j in range(batch)]
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        row = {k: sum(s[k] for s in stats) / batch for k in stats[0]}
        row["response_tokens"] = sum(s["response_tokens"] for s in stats)
        row.update(step=step+1, grad_norm=norm.item(), seconds=time.perf_counter()-start)
        if model.device.type == "cuda":
            row["peak_mib"] = torch.cuda.max_memory_allocated(model.device) / 2**20
        log.append(row)
        if (step+1) % cfg["save_every"] == 0 or step+1 == cfg["steps"]:
            state = dict(adapter=adapter_state(model), optimizer=optimizer.state_dict(), log=log)
            atomic_write(path, lambda tmp: torch.save(state, tmp))
            print(arm, {k: round(v, 4) for k, v in row.items()}, flush=True)
    optimizer.zero_grad(set_to_none=True)
    return log
