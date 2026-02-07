import os
import json
import string
import random
import numpy as np
import collections
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
    seq_len: int = 128
    num_train: int = 10000
    num_test: int = 1000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 6
    branch_prob: float = 0.5
    num_distractors: int = 4
    seed: int = 42

def compute_reachable(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> Set[str]:
    """
    Compute all facts reachable from start_facts using the given rules.
    This is critical for ensuring distractors don't accidentally fire.
    """
    known = set(start_facts)
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
                known.add(target)
                changed = True
    
    return known

def get_stratified_target(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> List[str]:
    """
    Simulates the logic flow to assign a DEPTH to each derived fact.
    Returns facts sorted by (Depth, Name).
    """
    known = compute_depths(start_facts, rules)
    derived = [(depth, target) for target, depth in known.items() if target not in start_facts]
    derived.sort(key=lambda x: (x[0], x[1]))
    return [x[1] for x in derived]

def compute_depths(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> Dict[str, int]:
    """
    Computes depth for all reachable facts.
    """
    known = {f: 0 for f in start_facts}  # Fact -> Depth
    changed = True
    max_iterations = 100
    iteration = 0
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        step_discoveries = {} 
        for premises, target in rules:
            if target in known: continue
            if all(p in known for p in premises):
                new_depth = max(known[p] for p in premises) + 1
                if target not in step_discoveries or new_depth < step_discoveries[target]:
                    step_discoveries[target] = new_depth
        if step_discoveries:
            changed = True
            for target, depth in step_discoveries.items():
                known[target] = depth
    return known

def generate_branching_sample(config: DataProcessConfig) -> Tuple[str, str]:
    available_vars = VARS[:config.num_vars]
    
    # 1. Initialize State
    num_start = random.randint(2, 3)
    known_facts = set(random.sample(available_vars, num_start))
    start_facts_list = sorted(list(known_facts))
    
    used_vars = set(known_facts)
    true_rules = []
    
    # 2. Build the Derivation Graph
    num_layers = random.randint(config.min_layers, config.max_layers)
    
    for _ in range(num_layers):
        potential_targets = [v for v in available_vars if v not in used_vars]
        if not potential_targets: break
        target = random.choice(potential_targets)
        is_branching = random.random() < config.branch_prob
        
        if is_branching and len(known_facts) >= 2:
            premises = random.sample(list(known_facts), 2)
        else:
            premises = random.sample(list(known_facts), 1)
            
        true_rules.append((premises, target))
        known_facts.add(target)
        used_vars.add(target)

    # 3. Reachable calculation
    reachable = compute_reachable(set(start_facts_list), true_rules)
    unused_vars = [v for v in available_vars if v not in reachable]
    
    # 4. Add Distractors (Simplified: Harmless + Isolated only)
    distractor_rules = []
    attempts = 0
    num_harmless = config.num_distractors // 2
    num_isolated = config.num_distractors - num_harmless
    true_depths = compute_depths(set(start_facts_list), true_rules)

    # TYPE 1: Harmless Reachable
    for _ in range(num_harmless):
        if len(reachable) < 2:
             num_isolated += 1
             continue
        attempts += 1
        if attempts > 500: break
        s_count = 2 if random.random() < config.branch_prob else 1
        premises = random.sample(list(reachable), min(s_count, len(reachable)))
        target = random.choice(list(reachable))
        # Avoid true rules or existing distractors
        if any((set(premises) == set(tr[0]) and target == tr[1]) for tr in true_rules + distractor_rules): continue
        # No shortcuts
        if all(p in true_depths for p in premises) and target in true_depths:
             new_path_depth = max(true_depths[p] for p in premises) + 1
             if new_path_depth < true_depths[target]: continue 
        distractor_rules.append((premises, target))

    # TYPE 2: Isolated
    for _ in range(num_isolated):
        if len(unused_vars) < 2: break
        attempts += 1
        if attempts > 500: break
        s_count = 2 if random.random() < config.branch_prob else 1
        premises = random.sample(unused_vars, min(s_count, len(unused_vars)))
        target = random.choice(available_vars) 
        if any((set(premises) == set(tr[0]) and target == tr[1]) for tr in true_rules + distractor_rules): continue
        distractor_rules.append((premises, target))

    # 5. Format Output (Commas)
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for premises, target in all_rules:
        lhs = "&".join(sorted(premises)) 
        rule_strs.append(f"{lhs}>{target}")
    
    sorted_ground_truth = get_stratified_target(set(start_facts_list), true_rules)
    
    # Check
    validation_target = get_stratified_target(set(start_facts_list), all_rules)
    if validation_target != sorted_ground_truth:
        raise ValueError(f"VALIDATION FAILED: Distractor changed output!")

    input_str = f"Facts: {' '.join(start_facts_list)} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(sorted_ground_truth)
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
                if failed_validations > 100:
                    raise RuntimeError(f"Too many validation failures ({failed_validations}). Check distractor logic.")
                continue
                
            full_text = f"{inp_str} {targ_str}"
            
            input_tokens = tokenize(inp_str, config.seq_len)
            label_tokens = tokenize(full_text, config.seq_len)
            
            if (input_tokens is None) or (label_tokens is None):
                continue
                
            # results["inputs"].append(np.array(input_tokens)) <--- REMOVED
            # results["labels"].append(np.array(label_tokens)) <--- REMOVED
            
            example_id += 1
            puzzle_id += 1
            results["puzzle_identifiers"].append(valid_count)
            results["group_indices"].append(puzzle_id)
            
            # Masking: Find "Target:" token (VOCAB['Target:']) and mask everything before it (EXCLUSIVE)
            # We keep the "Target:" token so the model knows when to start predicting.
            target_token_id = VOCAB['Target:']
            try:
                target_idx = label_tokens.index(target_token_id)
                for i in range(target_idx): # Exclusive
                    label_tokens[i] = VOCAB['pad']
            except ValueError:
                pass

            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))
            
            valid_count += 1
            pbar.update(1)
    
    if failed_validations > 0:
        print(f"Note: {failed_validations} samples failed validation and were regenerated.")

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
    print(f"Generating Branching Logic (Fixed & Aligned) in: {config.output_dir}")
    print(f"Config: layers={config.min_layers}-{config.max_layers}, "
          f"branch_prob={config.branch_prob}, distractors={config.num_distractors}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)
    print("Done.")

if __name__ == "__main__":
    cli()