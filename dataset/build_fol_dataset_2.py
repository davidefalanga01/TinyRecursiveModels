#!/usr/bin/env python3
import os
import json
import string
import random
import argparse
from dataclasses import dataclass
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
from tqdm import tqdm


# ----------------------------
# Config
# ----------------------------

@dataclass
class DataProcessConfig:
    output_dir: str = "data/logic_chain"
    seq_len: int = 160
    num_train: int = 20000
    num_test: int = 2000
    num_vars: int = 26
    min_steps: int = 2
    max_steps: int = 6
    num_distractors: int = 4
    seed: int = 42
    max_resample: int = 200


VARS = list(string.ascii_uppercase)

SPECIALS = [
    "Facts:", "Rules:", "Target:",
    "|", ">",
]

VOCAB: Dict[str, int] = {
    "pad": 0,
    "end": 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)},
}

PAD_ID = VOCAB["pad"]
END_ID = VOCAB["end"]


# ----------------------------
# Logic
# ----------------------------

Rule = Tuple[str, str]


def forward_min_depth(start_facts: Set[str], rules: List[Rule]) -> Dict[str, int]:
    depth: Dict[str, int] = {f: 0 for f in start_facts}
    
    changed = True
    for _ in range(64):
        if not changed:
            break
        changed = False
        for s, t in rules:
            if s not in depth:
                continue
            cand = depth[s] + 1
            if t not in depth or cand < depth[t]:
                depth[t] = cand
                changed = True
    return depth


def derived_sequence(start_facts: Set[str], rules: List[Rule]) -> List[str]:
    depth = forward_min_depth(start_facts, rules)
    derived = [(d, v) for v, d in depth.items() if v not in start_facts]
    derived.sort(key=lambda x: (x[0], x[1]))
    return [v for _, v in derived]


# ----------------------------
# Text formatting + tokenization
# ----------------------------

def format_problem(start_facts: Set[str], rules: List[Rule]) -> str:
    sf = " ".join(sorted(start_facts))
    rules_str = " ".join([f"{s}>{t}" for s, t in rules])
    return f"Facts: {sf} | Rules: {rules_str} | Target:"


def format_target(seq: List[str]) -> str:
    return " ".join(seq)


def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        matched_special = False
        for special in sorted(SPECIALS, key=len, reverse=True): 
            if text[i:].startswith(special):
                tokens.append(VOCAB[special])
                i += len(special)
                matched_special = True
                break
        if matched_special: continue
        if c in VARS:
            tokens.append(VOCAB[c])
            i += 1
            continue
        i += 1 
    if len(tokens) > seq_len - 1: return None
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens


# ----------------------------
# Dataset generation (FIXED)
# ----------------------------

def generate_chain_sample(cfg: DataProcessConfig) -> Tuple[str, str]:
    """
    FIXED VERSION: Distractors now properly isolated and won't confuse the model.
    """
    chain_len = random.randint(cfg.min_steps, cfg.max_steps)

    available = VARS[:cfg.num_vars]
    
    # Updated to match logic from newer datasets
    num_start = random.randint(1, 2)

    needed = num_start + chain_len
    if needed > len(available):
        available = VARS[:]

    chosen = random.sample(available, needed)
    start = set(chosen[:num_start])
    derived_vars = chosen[num_start:]

    # True rules: create chain
    true_rules: List[Rule] = []
    true_rules.append((random.choice(sorted(start)), derived_vars[0]))
    for k in range(chain_len - 1):
        true_rules.append((derived_vars[k], derived_vars[k + 1]))

    if num_start >= 2 and random.random() < 0.5:
        # Find a start fact that wasn't the primary trigger for d0
        primary_trigger = true_rules[0][0]
        other_starts = [x for x in start if x != primary_trigger]
        if other_starts:
            s2 = other_starts[0]
            true_rules.append((s2, derived_vars[0]))  # duplicate target, harmless

    # FIXED: Build truly isolated distractors
    used = set(chosen)
    unused = [v for v in available if v not in used]
    distractors: List[Rule] = []

    reachable = set(start) | set(derived_vars)

    # Type 1: reachable -> reachable (but NEVER self-loops, NEVER to start facts)
    derived_only = set(derived_vars)  # Only target derived facts
    for _ in range(cfg.num_distractors // 2):
        if len(derived_only) < 1:
            break
        s = random.choice(list(reachable))
        # CRITICAL FIX: Only target derived facts, never start facts
        t = random.choice(list(derived_only))
        # CRITICAL FIX: Prevent self-loops
        if s == t:
            continue
        distractors.append((s, t))

    # Type 2: completely isolated subgraph
    for _ in range(cfg.num_distractors - len(distractors)):
        if len(unused) < 2:
            break
        s, t = random.sample(unused, 2)
        # Still prevent self-loops even in isolated graphs
        if s == t:
            continue
        distractors.append((s, t))

    all_rules = true_rules + distractors
    random.shuffle(all_rules)

    target_seq = derived_sequence(start, true_rules)

    inp = format_problem(start, all_rules)
    targ = format_target(target_seq)
    return inp, targ


def convert_subset(set_name: str, cfg: DataProcessConfig, num_samples: int) -> None:
    inputs = []
    labels = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []

    valid = 0
    pbar = tqdm(total=num_samples, desc=f"Generating {set_name}")

    while valid < num_samples:
        ok = False
        for _ in range(cfg.max_resample):
            inp, targ = generate_chain_sample(cfg)

            full = f"{inp} {targ}"
            it = tokenize(inp, cfg.seq_len)
            ft = tokenize(full, cfg.seq_len)
            if it is None or ft is None:
                continue

            ok = True
            inputs.append(np.array(it, dtype=np.int32))
            labels.append(np.array(ft, dtype=np.int32))
            break

        if not ok:
            raise RuntimeError("Failed to generate a valid sample within max_resample attempts.")

        puzzle_identifiers.append(valid)
        valid += 1
        puzzle_indices.append(valid)
        group_indices.append(valid)
        pbar.update(1)

    pbar.close()

    def approx_target_len(lbl: np.ndarray) -> int:
        nz = int(np.count_nonzero(lbl != PAD_ID))
        return nz

    order = np.argsort([approx_target_len(x) for x in labels])
    inputs = np.stack(inputs, axis=0)[order]
    labels = np.stack(labels, axis=0)[order]

    puzzle_identifiers = np.array(puzzle_identifiers, dtype=np.int32)
    puzzle_indices = np.array(puzzle_indices, dtype=np.int32)
    group_indices = np.array(group_indices, dtype=np.int32)

    save_dir = os.path.join(cfg.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), puzzle_indices)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), group_indices)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), puzzle_identifiers)

    metadata = {
        "seq_len": cfg.seq_len,
        "vocab_size": len(VOCAB),
        "pad_id": PAD_ID,
        "ignore_label_id": PAD_ID,
        "blank_identifier_id": 0,
        "total_groups": int(len(group_indices) - 1),
        "mean_puzzle_examples": 1.0,
        "num_puzzle_identifiers": int(len(puzzle_identifiers)),
        "total_puzzles": int(len(puzzle_identifiers)),
        "sets": ["all"],
    }
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    if set_name == "train":
        with open(os.path.join(cfg.output_dir, "vocab.json"), "w") as f:
            json.dump(VOCAB, f)
        with open(os.path.join(cfg.output_dir, "identifiers.json"), "w") as f:
            json.dump([f"logic_chain_{i}" for i in range(int(len(puzzle_identifiers)))], f)

        example_inp, example_targ = generate_chain_sample(cfg)
        print("\n" + "=" * 70)
        print("EXAMPLE (human-readable):")
        print(example_inp)
        print("TARGET:", example_targ)
        print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser("Generate implication-chain dataset (FIXED v3)")
    p.add_argument("--output-dir", type=str, default=DataProcessConfig.output_dir)
    p.add_argument("--seq-len", type=int, default=DataProcessConfig.seq_len)
    p.add_argument("--num-train", type=int, default=DataProcessConfig.num_train)
    p.add_argument("--num-test", type=int, default=DataProcessConfig.num_test)
    p.add_argument("--num-vars", type=int, default=DataProcessConfig.num_vars)
    p.add_argument("--min-steps", type=int, default=DataProcessConfig.min_steps)
    p.add_argument("--max-steps", type=int, default=DataProcessConfig.max_steps)
    p.add_argument("--num-distractors", type=int, default=DataProcessConfig.num_distractors)
    p.add_argument("--seed", type=int, default=DataProcessConfig.seed)
    args = p.parse_args()

    cfg = DataProcessConfig(
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        num_train=args.num_train,
        num_test=args.num_test,
        num_vars=args.num_vars,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        num_distractors=args.num_distractors,
        seed=args.seed,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Generating implication-chain dataset (FIXED v3).")
    print(f"Output: {cfg.output_dir}, seq_len={cfg.seq_len}, train={cfg.num_train}, test={cfg.num_test}")
    convert_subset("train", cfg, cfg.num_train)
    convert_subset("test", cfg, cfg.num_test)
    print("Done.")


if __name__ == "__main__":
    main()