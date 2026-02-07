import os
import json
import string
import random
import numpy as np
from typing import Optional, List, Tuple, Set, Dict
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '|']

VOCAB = {
    'pad': 0,
    'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_branching"
    seq_len: int = 160  # Fixed: removed duplicate
    num_train: int = 20000
    num_test: int = 2000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 6
    branch_prob: float = 0.5
    num_distractors: int = 4
    seed: int = 42

def compute_depths(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> Dict[str, int]:
    """
    Computes depth for all reachable facts.
    Depth = maximum depth of premises + 1
    """
    known = {f: 0 for f in start_facts}
    changed = True
    max_iterations = 100
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for premises, target in rules:
            if target in known:
                continue
            if all(p in known for p in premises):
                new_depth = max(known[p] for p in premises) + 1
                known[target] = new_depth
                changed = True
    
    return known

def get_stratified_target(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> List[str]:
    """
    Returns derived facts sorted by (Depth, Name).
    """
    depths = compute_depths(start_facts, rules)
    derived = [(depth, var) for var, depth in depths.items() if var not in start_facts]
    derived.sort(key=lambda x: (x[0], x[1]))
    return [x[1] for x in derived]

def generate_branching_sample(config: DataProcessConfig) -> Tuple[str, str]:
    available_vars = VARS[:config.num_vars]

    # 1. Initialize with start facts
    num_start = random.randint(1, 2)
    start_facts = set(random.sample(available_vars, num_start))
    start_facts_list = sorted(list(start_facts))

    known_facts = set(start_facts)
    used_vars = set(start_facts)
    true_rules = []

    # 2. Build derivation graph
    num_layers = random.randint(config.min_layers, config.max_layers)

    for _ in range(num_layers):
        potential_targets = [v for v in available_vars if v not in used_vars]
        if not potential_targets:
            break
        
        target = random.choice(potential_targets)
        is_branching = random.random() < config.branch_prob

        if is_branching and len(known_facts) >= 2:
            premises = random.sample(list(known_facts), 2)
        else:
            premises = random.sample(list(known_facts), 1)

        true_rules.append((premises, target))
        known_facts.add(target)
        used_vars.add(target)

    # 3. Compute reachable set and depths
    true_depths = compute_depths(start_facts, true_rules)
    reachable = set(true_depths.keys())
    derived_facts = reachable - start_facts
    unused_vars = [v for v in available_vars if v not in reachable]

    # 4. Generate Distractors
    distractor_rules = []
    
    # TYPE 1: Harmless (reachable -> reachable, but doesn't change depths)
    # Strategy: Only allow rules where target is ALREADY reachable at same or shallower depth
    num_harmless = config.num_distractors // 2
    
    attempts = 0
    while len([d for d in distractor_rules if d[0]]) < num_harmless and attempts < 500:
        attempts += 1
        
        if len(reachable) < 2:
            break
        
        # Pick premises and target from reachable
        is_branching = random.random() < config.branch_prob
        s_count = 2 if is_branching and len(reachable) >= 2 else 1
        premises = random.sample(list(reachable), min(s_count, len(reachable)))
        
        # CRITICAL FIX: Only target derived facts (not start facts)
        if not derived_facts:
            break
        target = random.choice(list(derived_facts))
        
        # Avoid duplicates
        if (set(premises), target) in [(set(r[0]), r[1]) for r in true_rules + distractor_rules]:
            continue
        
        # CRITICAL FIX: Only allow if this rule would NOT change the depth
        # The new depth would be max(premise_depths) + 1
        new_depth = max(true_depths[p] for p in premises) + 1
        actual_depth = true_depths[target]
        
        # Allow only if new path is LONGER or EQUAL (won't fire first)
        if new_depth >= actual_depth:
            distractor_rules.append((premises, target))
    
    # TYPE 2: Isolated (completely unreachable)
    num_isolated = config.num_distractors - len(distractor_rules)
    
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 500:
        attempts += 1
        
        if len(unused_vars) < 2:
            break
        
        is_branching = random.random() < config.branch_prob
        s_count = 2 if is_branching and len(unused_vars) >= 2 else 1
        premises = random.sample(unused_vars, min(s_count, len(unused_vars)))
        
        # CRITICAL FIX: Target must also be from unused (stay isolated)
        target = random.choice(unused_vars)
        
        # Avoid duplicates
        if (set(premises), target) in [(set(r[0]), r[1]) for r in true_rules + distractor_rules]:
            continue
        
        distractor_rules.append((premises, target))

    # 5. Validate distractors don't change output
    all_rules = true_rules + distractor_rules
    validation_target = get_stratified_target(start_facts, all_rules)
    expected_target = get_stratified_target(start_facts, true_rules)
    
    if validation_target != expected_target:
        raise ValueError(
            f"VALIDATION FAILED!\n"
            f"Expected: {expected_target}\n"
            f"Got: {validation_target}\n"
            f"Start: {start_facts_list}\n"
            f"True rules: {true_rules}\n"
            f"Distractors: {distractor_rules}"
        )

    # 6. Format output
    random.shuffle(all_rules)
    
    rule_strs = []
    for premises, target in all_rules:
        lhs = "&".join(sorted(premises))
        rule_strs.append(f"{lhs}>{target}")

    input_str = f"Facts: {' '.join(start_facts_list)} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(expected_target)
    
    return input_str, target_str

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
        
        if matched_special:
            continue
        
        if c in VARS:
            tokens.append(VOCAB[c])
            i += 1
            continue
        
        i += 1
    
    if len(tokens) > seq_len - 1:
        return None
    
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens

def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    valid_count = 0
    failed_validations = 0

    with tqdm(total=num_samples, desc=f"Generating {set_name}") as pbar:
        while valid_count < num_samples:
            try:
                inp_str, targ_str = generate_branching_sample(config)
            except ValueError as e:
                failed_validations += 1
                if failed_validations > 1000:
                    print(f"\nERROR: Too many validation failures. Last error:\n{e}")
                    raise RuntimeError(f"Distractor generation is broken. Check logic.")
                continue

            full_text = f"{inp_str} {targ_str}"

            input_tokens = tokenize(inp_str, config.seq_len)
            label_tokens = tokenize(full_text, config.seq_len)

            if (input_tokens is None) or (label_tokens is None):
                continue

            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))

            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(valid_count)
            results["group_indices"].append(puzzle_id)

            valid_count += 1
            pbar.update(1)

    if failed_validations > 0:
        print(f"\nNote: {failed_validations} samples failed validation and were regenerated.")

    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = {
        "seq_len": config.seq_len,
        "vocab_size": len(VOCAB),
        "pad_id": VOCAB['pad'],
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": valid_count,
        "total_groups": len(final_results["group_indices"]) - 1,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": puzzle_id,
        "sets": ["all"]
    }

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    if set_name == "train":
        with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
            json.dump([f"logic_{i}" for i in range(valid_count)], f)
        with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
            json.dump(VOCAB, f)

        # Print example
        print("\n" + "=" * 70)
        print("EXAMPLE (human-readable):")
        example_inp, example_targ = generate_branching_sample(config)
        print(example_inp)
        print("TARGET:", example_targ)
        print("=" * 70 + "\n")

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    print(f"Generating Branching Logic in: {config.output_dir}")
    print(f"Config: layers={config.min_layers}-{config.max_layers}, "
          f"branch_prob={config.branch_prob}, distractors={config.num_distractors}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)
    print("Done.")

if __name__ == "__main__":
    cli()  # FIXED: Added missing call