"""Paired, verifiable evidence tasks and the two evaluation prompt formats."""
import random
import re

ABSTAIN = "INSUFFICIENT"
FAMILIES = {
    "rail": ("Train", "platform"),
    "venue": ("Event", "room"),
    "inventory": ("Parcel", "bin"),
}
STRICT = "Return only the requested code, or INSUFFICIENT if the records do not determine it."
NATURAL = "Answer briefly in ordinary language. If the records do not determine the answer, explain why."
CODE_PATTERN = r"[A-H][0-9]"


def make_cases(split, pairs, seed):
    rng = random.Random(f"{seed}/{split}")
    rows = []
    for i in range(pairs):
        family = "inventory" if split == "test" else ("rail", "venue")[i % 2]
        noun, field = FAMILIES[family]
        ids = rng.sample(range(1000, 10000), 6)
        codes = rng.sample([f"{letter}{digit}" for letter in "ABCDEFGH" for digit in range(10)], 5)
        linked = (i // 2) % 2 == 1
        groups = rng.sample(range(100, 1000), 6)
        first_order = rng.sample(range(5), 5)
        last_order = rng.sample(range(5), 5)
        for answerable in [True, False]:
            keys = ids[:5]
            lookup = groups[:5]
            if not answerable:
                if linked and (i // 4) % 2 == 0:
                    lookup = [groups[5], *groups[1:5]]
                else:
                    keys = [ids[5], *ids[1:5]]
            if linked:
                records = [f"{noun} {keys[j]} | group {groups[j]}" for j in first_order]
                records += [f"Group {lookup[j]} | {field} {codes[j]}" for j in last_order]
            else:
                records = [f"{noun} {keys[j]} | {field} {codes[j]}" for j in first_order]
            question = f"Which {field} is assigned to {noun.lower()} {ids[0]}?"
            task = "\n".join(records) + "\n\n" + question
            rows.append(dict(pair=f"{split}-{i}", family=family, linked=linked, answerable=answerable,
                             task=task, target=codes[0] if answerable else ABSTAIN))
    return rows


def instruction(row, natural=False):
    return f"Use only these records. {NATURAL if natural else STRICT}\n\n{row['task']}"


def grade(answer, row):
    # Accept one unambiguous code, optionally introduced by the correct field.
    value = answer.strip().upper().removesuffix(".")
    field = FAMILIES[row["family"]][1].upper()
    match = re.fullmatch(rf"(?:{field}\s+)?({CODE_PATTERN})", value)
    if match:
        value = match[1]
    valid = value == ABSTAIN or match is not None
    return {"correct": int(value == row["target"]), "abstained": int(value == ABSTAIN), "invalid": int(not valid)}
