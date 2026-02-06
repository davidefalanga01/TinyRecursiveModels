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
    seq_len: int = 256
    num_train: int = 20000
    num_test: int = 2000
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
    Uses synchronous updates (like BFS) to ensure shortest derivation path is used for depth.
    """
    known = compute_depths(start_facts, rules)
    
    # Sort by Depth first, then Alphabetical
    derived = [(depth, target) for target, depth in known.items() if target not in start_facts]
    derived.sort(key=lambda x: (x[0], x[1]))
    
    return [x[1] for x in derived]

def compute_depths(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> Dict[str, int]:
    """
    Computes depth for all reachable facts.
    Returns Dictionary {fact: depth}
    """
    known = {f: 0 for f in start_facts}  # Fact -> Depth

    changed = True
    max_iterations = 100
    iteration = 0
        
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # Collect all discoveries for this step first
        step_discoveries = {} # target -> depth
        
        for premises, target in rules:
            if target in known:
                continue
            
            # Check if all premises are known from PREVIOUS steps
            if all(p in known for p in premises):
                # Depth = max(premise_depths) + 1
                new_depth = max(known[p] for p in premises) + 1
                
                # If we found multiple paths to the same target in this step,
                # keep the one with the smallest depth (shortest path)
                if target not in step_discoveries or new_depth < step_discoveries[target]:
                    step_discoveries[target] = new_depth
        
        # Apply updates
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

    # 3. CRITICAL FIX: Compute reachable set before generating distractors
    reachable = compute_reachable(set(start_facts_list), true_rules)
    unused_vars = [v for v in available_vars if v not in reachable]
    
    # 4. Add Distractors with SAFE generation strategy
    distractor_rules = []
    attempts = 0
    
    # Target distribution: 1/3 each type
    num_type1 = config.num_distractors // 3
    num_type2 = config.num_distractors // 3
    num_type3 = config.num_distractors - num_type1 - num_type2
    
    # TYPE 1: Missing one critical premise (tests partial matching)
    # One premise is reachable, one is NOT → rule never fires
    for _ in range(num_type1):
        if len(reachable) < 1 or len(unused_vars) < 1:
            break
        attempts += 1
        if attempts > 500:
            break
            
        good_premise = random.choice(list(reachable))
        bad_premise = random.choice(unused_vars)
        
        # Mix them so order doesn't give away which is bad
        premises = [good_premise, bad_premise]
        random.shuffle(premises)
        
        target = random.choice(available_vars)
        
        # Avoid duplicates
        if any((set(premises) == set(tr[0]) and target == tr[1]) for tr in true_rules + distractor_rules):
            continue
            
        distractor_rules.append((premises, target))
    
    # TYPE 2: All unreachable premises (never fires, isolated subgraph)
    for _ in range(num_type2):
        if len(unused_vars) < 2:
            break
        attempts += 1
        if attempts > 500:
            break
            
        s_count = 2 if random.random() < config.branch_prob else 1
        premises = random.sample(unused_vars, min(s_count, len(unused_vars)))
        target = random.choice(available_vars)
        
        # Avoid duplicates
        if any((set(premises) == set(tr[0]) and target == tr[1]) for tr in true_rules + distractor_rules):
            continue
            
        distractor_rules.append((premises, target))
    
    # TYPE 3: Reachable -> Reachable (harmless, target already derived)
    # These rules CAN fire, but they don't add new facts
    # CRITICAL FIX: Ensure they don't provide a SHORTCUT to the target!
    
    # Compute original depths to check for shortcuts
    true_depths = compute_depths(set(start_facts_list), true_rules)
    
    for _ in range(num_type3):
        if len(reachable) < 2:
            break
        attempts += 1
        if attempts > 500:
            break
            
        s_count = 2 if random.random() < config.branch_prob else 1
        premises = random.sample(list(reachable), min(s_count, len(reachable)))
        target = random.choice(list(reachable))
        
        # Avoid duplicates
        if any((set(premises) == set(tr[0]) and target == tr[1]) for tr in true_rules + distractor_rules):
            continue
            
        # CHECK FOR SHORTCUTS
        # New depth would be max(premise_depths) + 1
        # If this is < original_depth, it's a shortcut -> REJECT
        if all(p in true_depths for p in premises) and target in true_depths:
             new_path_depth = max(true_depths[p] for p in premises) + 1
             if new_path_depth < true_depths[target]:
                 # This rule creates a faster way to get to target -> Bad distractor
                 continue
            
        distractor_rules.append((premises, target))

    # 5. Format Output with Stratified Sorting
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for premises, target in all_rules:
        # Sort premises for canonical representation
        lhs = "&".join(sorted(premises)) 
        rule_strs.append(f"{lhs}>{target}")
    
    # Calculate the Stratified Target from TRUE rules only
    sorted_ground_truth = get_stratified_target(set(start_facts_list), true_rules)
    
    # 6. VALIDATION: Ensure distractors don't change output
    # This catches any bugs in distractor generation
    validation_target = get_stratified_target(set(start_facts_list), all_rules)
    if validation_target != sorted_ground_truth:
        # This should NEVER happen with correct distractor generation
        # If it does, it means a distractor fired and added new facts
        raise ValueError(
            f"VALIDATION FAILED: Distractor changed output!\n"
            f"Expected: {sorted_ground_truth}\n"
            f"Got: {validation_target}\n"
            f"Start: {start_facts_list}\n"
            f"True rules: {true_rules}\n"
            f"Distractors: {distractor_rules}"
        )

    input_str = f"Facts: {' '.join(start_facts_list)} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(sorted_ground_truth)
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    raw_parts = text.split(' ')
    
    for part in raw_parts:
        if not part:
            continue
        if part in VOCAB:
            tokens.append(VOCAB[part])
            continue
            
        buffer = ""
        for char in part:
            if char in ['&', '>', '|']:
                if buffer and buffer in VOCAB:
                    tokens.append(VOCAB[buffer])
                elif buffer:
                    # Unknown token - skip with warning in debug mode
                    pass
                tokens.append(VOCAB[char])
                buffer = ""
            else:
                buffer += char
        if buffer and buffer in VOCAB:
            tokens.append(VOCAB[buffer])

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
                # Validation failed - this sample has a bug
                failed_validations += 1
                if failed_validations > 100:
                    raise RuntimeError(f"Too many validation failures ({failed_validations}). Check distractor logic.")
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
    print(f"Generating Branching Logic (Fixed) in: {config.output_dir}")
    print(f"Config: layers={config.min_layers}-{config.max_layers}, "
          f"branch_prob={config.branch_prob}, distractors={config.num_distractors}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)
    print("Done.")

if __name__ == "__main__":
    cli()