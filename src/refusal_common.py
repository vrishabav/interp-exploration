"""Qwen prompt formatting, hook lifetime and controls shared by both notebooks."""
from contextlib import contextmanager

import numpy as np

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EOI = "<|im_end|>\n<|im_start|>assistant\n"


def local_model_source(root, model_name):
    local = root / "cache" / "rl_models" / QWEN_REVISION
    if model_name == QWEN_MODEL and (local / "model.safetensors").is_file():
        return local
    return model_name


def format_instruction(instruction, system_prompt=None):
    prefix = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    return f"{prefix}<|im_start|>user\n{instruction}{EOI}"


@contextmanager
def hooks(fwd_pre=(), fwd=()):
    handles = []
    try:
        for module, fn in fwd_pre:
            handles.append(module.register_forward_pre_hook(fn, with_kwargs=True))
        for module, fn in fwd:
            handles.append(module.register_forward_hook(fn))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def random_controls(neurons, seed, width, excluded):
    rng = np.random.default_rng(seed)
    taken = set(excluded)
    result = []
    for layer, _ in neurons:
        while True:
            neuron = (layer, int(rng.integers(width)))
            if neuron not in taken:
                taken.add(neuron)
                result.append(neuron)
                break
    return result
