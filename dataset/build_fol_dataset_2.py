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
    min_steps: int = 2          # number of derived steps
    max_steps: int = 6
    num_distractors: int = 4    # total distractor rules added to input
    multi_start: bool = True    # allow 2 start facts sometimes
    seed: int = 42
    max_resample: int = 200     # attempts per example


VARS = list(string.ascii_uppercase)

# Tokens used in text format
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

Rule = Tuple[str, str]  # (source, target), single-premise implication


def forward_min_depth(start_facts: Set[str], rules: List[Rule]) -> Dict[str, int]:
    """
    Compute minimal derivation depth for each reachable fact under single-premise rules.
    Depth(start facts)=0, depth(derived)=min depth along any path.
    """
    depth: Dict[str, int] = {f: 0 for f in start_facts}

    changed = True
    # For these tiny graphs, simple relaxation is fine and deterministic with sorting.
    # (We also do not allow negative premises here.)
    for _ in range(64):  # hard cap; should converge much earlier
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
    """
    Return derived facts (excluding start facts) sorted by (depth, name).
    """
    depth = forward_min_depth(start_facts, rules)
    derived = [(d, v) for v, d in depth.items() if v not in start_facts]
    derived.sort(key=lambda x: (x[0], x[1]))
    return [v for _, v in derived]


# ----------------------------
# Text formatting + tokenization
# ----------------------------

def format_problem(start_facts: Set[str], rules: List[Rule]) -> str:
    sf = " ".join(sorted(start_facts))
    # IMPORTANT: put spaces around punctuation so the string is readable,
    # but tokenizer below is robust even if you later change formatting.
    rules_str = " ".join([f"{s} > {t}" for s, t in rules])
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
        # Unknown char? -> skip or treat as error. 
        # The other script simply ignores 'c' if not matched and increments i? 
        # Wait, the other script has "i += 1" at the end of loop.
        i += 1 
    if len(tokens) > seq_len - 1: return None
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens


# ----------------------------
# Dataset generation
# ----------------------------

def generate_chain_sample(cfg: DataProcessConfig) -> Tuple[str, str]:
    """
    Builds:
    - a true reachable chain of length L (L derived facts),
    - distractors that are guaranteed to NOT add new derived facts beyond the true chain.
    """
    chain_len = random.randint(cfg.min_steps, cfg.max_steps)

    available = VARS[:cfg.num_vars]
    if cfg.multi_start and chain_len >= 3:
        num_start = random.randint(1, 2)
    else:
        num_start = 1

    # Choose variables: start facts + chain_len derived
    needed = num_start + chain_len
    if needed > len(available):
        available = VARS[:]  # fallback

    chosen = random.sample(available, needed)
    start = set(chosen[:num_start])
    derived_vars = chosen[num_start:]  # length = chain_len

    # True rules: start -> d0, d0 -> d1 -> ...
    true_rules: List[Rule] = []
    true_rules.append((random.choice(sorted(start)), derived_vars[0]))
    for k in range(chain_len - 1):
        true_rules.append((derived_vars[k], derived_vars[k + 1]))

    # Optionally make 2nd start fact relevant but harmless (doesn't change targets)
    # by pointing it to an already-derived var (fires but adds nothing new).
    if num_start == 2 and random.random() < 0.5:
        s2 = [x for x in start if x != true_rules[0][0]][0]
        true_rules.append((s2, derived_vars[0]))  # duplicate target, harmless

    # Build distractors that do NOT affect derivation:
    used = set(chosen)
    unused = [v for v in available if v not in used]
    distractors: List[Rule] = []

    reachable = set(start) | set(derived_vars)

    # Type 1 distractors: reachable -> reachable (target already reachable => no new facts)
    # (These may fire, but they never add a new symbol.)
    for _ in range(cfg.num_distractors // 2):
        s = random.choice(list(reachable))
        t = random.choice(list(reachable))
        distractors.append((s, t))

    # Type 2 distractors: isolated subgraph among unused vars only (unreachable forever)
    # Ensure we never connect from reachable into unused.
    for _ in range(cfg.num_distractors - len(distractors)):
        if len(unused) < 2:
            break
        s, t = random.sample(unused, 2)
        distractors.append((s, t))

    # Shuffle rules for input presentation
    all_rules = true_rules + distractors
    random.shuffle(all_rules)

    # Ground truth must be computed from TRUE rules only
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

            # Sanity check: distractors should not change the target set.
            # We re-parse by simulation on true rules only, so this should always pass,
            # but it also catches accidental formatting/token mistakes if you edit later.
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

    # Curriculum: sort by target length (number of derived facts)
    def approx_target_len(lbl: np.ndarray) -> int:
        # count non-pad up to end
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

        # Print one example after sorting (easy sample)
        example_inp = None
        example_targ = None
        # regenerate a readable example (not from tokens) for display
        example_inp, example_targ = generate_chain_sample(cfg)
        print("\n" + "=" * 70)
        print("EXAMPLE (human-readable):")
        print(example_inp)
        print("TARGET:", example_targ)
        print("=" * 70 + "\n")


def main():
    p = argparse.ArgumentParser("Generate implication-chain dataset (fixed v2)")
    p.add_argument("--output-dir", type=str, default=DataProcessConfig.output_dir)
    p.add_argument("--seq-len", type=int, default=DataProcessConfig.seq_len)
    p.add_argument("--num-train", type=int, default=DataProcessConfig.num_train)
    p.add_argument("--num-test", type=int, default=DataProcessConfig.num_test)
    p.add_argument("--num-vars", type=int, default=DataProcessConfig.num_vars)
    p.add_argument("--min-steps", type=int, default=DataProcessConfig.min_steps)
    p.add_argument("--max-steps", type=int, default=DataProcessConfig.max_steps)
    p.add_argument("--num-distractors", type=int, default=DataProcessConfig.num_distractors)
    p.add_argument("--multi-start", action="store_true")
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
        multi_start=args.multi_start,
        seed=args.seed,
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    print("Generating implication-chain dataset (v2 fixed).")
    print(f"Output: {cfg.output_dir}, seq_len={cfg.seq_len}, train={cfg.num_train}, test={cfg.num_test}")
    convert_subset("train", cfg, cfg.num_train)
    convert_subset("test", cfg, cfg.num_test)
    print("Done.")


if __name__ == "__main__":
    main()
