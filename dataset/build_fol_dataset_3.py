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
    num_train: int = 50000
    num_test: int = 5000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 6
    branch_prob: float = 0.5
    num_distractors: int = 4
    seed: int = 42

def get_stratified_target(start_facts: Set[str], rules: List[Tuple[List[str], str]]) -> List[str]:
    """
    Simulates the logic flow to assign a DEPTH to each derived fact.
    Returns facts sorted by (Depth, Name).
    """
    known = {f: 0 for f in start_facts} # Fact -> Depth
    derived = []
    
    # Simple forward chaining to determine depth
    changed = True
    while changed:
        changed = False
        # Iterate over rules to see what activates
        for premises, target in rules:
            if target in known: continue
            
            # Check if all premises are known
            if all(p in known for p in premises):
                # Depth = max(premise_depths) + 1
                new_depth = max(known[p] for p in premises) + 1
                known[target] = new_depth
                derived.append((new_depth, target))
                changed = True
    
    # Sort by Depth first, then Alphabetical
    # x[0] is depth, x[1] is name
    derived.sort(key=lambda x: (x[0], x[1]))
    
    return [x[1] for x in derived]

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

    # 3. Add Distractors
    distractor_rules = []
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 200:
        attempts += 1
        s_count = 2 if random.random() < config.branch_prob else 1
        s = random.sample(available_vars, s_count)
        t = random.choice(available_vars)
        
        # Validation: Don't replicate existing, don't accidentally fire
        if any((set(s) == set(tr[0]) and t == tr[1]) for tr in true_rules + distractor_rules):
            continue
        if set(s).issubset(known_facts):
            continue
            
        distractor_rules.append((s, t))

    # 4. Format Output with Stratified Sorting
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for premises, target in all_rules:
        # Sort premises for canonical representation (optional but clean)
        lhs = "&".join(sorted(premises)) 
        rule_strs.append(f"{lhs}>{target}")
    
    # Calculate the Stratified Target
    sorted_ground_truth = get_stratified_target(set(start_facts_list), true_rules)

    input_str = f"Facts: {' '.join(start_facts_list)} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(sorted_ground_truth)
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    raw_parts = text.split(' ')
    
    for part in raw_parts:
        if not part: continue
        if part in VOCAB:
            tokens.append(VOCAB[part])
            continue
        buffer = ""
        for char in part:
            if char in ['&', '>', '|']:
                if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])
                tokens.append(VOCAB[char])
                buffer = ""
            else:
                buffer += char
        if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])

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
    with tqdm(total=num_samples, desc=f"Generating {set_name}") as pbar:
        while valid_count < num_samples:
            inp_str, targ_str = generate_branching_sample(config)
            full_text = f"{inp_str} {targ_str}"
            
            input_tokens = tokenize(inp_str, config.seq_len)
            label_tokens = tokenize(full_text, config.seq_len)
            
            if (input_tokens is None) or (label_tokens is None): continue
                
            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))
            
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(valid_count)
            results["group_indices"].append(puzzle_id)
            
            valid_count += 1
            pbar.update(1)

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
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([f"logic_{i}" for i in range(valid_count)], f)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(VOCAB, f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    print(f"Generating Branching Logic (Stratified) in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()