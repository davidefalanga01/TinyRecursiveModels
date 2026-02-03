import os
import json
import string
import random
import numpy as np
from typing import Optional, List, Tuple, Set
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# --- Vocabulary Definition ---
# Vars: A-Z
# Structure: Facts, Rules, Target, >, &, | (separator), end, pad
VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '|']

VOCAB = {
    'pad': 0,
    'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_branching"
    seq_len: int = 128          # Increased len (branching rules are longer)
    num_train: int = 50000
    num_test: int = 5000
    
    # Task Specific Configs
    num_vars: int = 26          # A-Z
    min_layers: int = 2         # Min depth of reasoning
    max_layers: int = 6         # Max depth
    branch_prob: float = 0.5    # Probability a rule is (A & B > C) vs (A > C)
    num_distractors: int = 4    # Fake rules
    seed: int = 42

def generate_branching_sample(config: DataProcessConfig) -> Tuple[str, str]:
    """
    Generates a Horn Clause derivation graph.
    Guarantees a deterministic path of discovery.
    """
    available_vars = VARS[:config.num_vars]
    
    # 1. Initialize State
    # Pick 2-3 start facts
    num_start = random.randint(2, 3)
    known_facts = set(random.sample(available_vars, num_start))
    start_facts_list = sorted(list(known_facts))
    
    # Track used variables to avoid re-discovering the same fact
    used_vars = set(known_facts)
    
    true_rules = []
    ground_truth_sequence = []
    
    # 2. Build the Derivation Chain (Layer by Layer)
    # We construct the "solution" forward, ensuring it's valid.
    num_layers = random.randint(config.min_layers, config.max_layers)
    
    for _ in range(num_layers):
        # We need to deduce a NEW variable
        # Potential targets are variables we haven't used yet
        potential_targets = [v for v in available_vars if v not in used_vars]
        if not potential_targets:
            break
            
        target = random.choice(potential_targets)
        
        # Decide if this rule is Simple (A > C) or Branching (A & B > C)
        # We must pick premises from 'known_facts'
        is_branching = random.random() < config.branch_prob
        
        # For branching, we need at least 2 known facts
        if is_branching and len(known_facts) >= 2:
            premises = random.sample(list(known_facts), 2)
        else:
            premises = random.sample(list(known_facts), 1)
            
        # Store the rule
        # Format: (['A', 'B'], 'C')
        true_rules.append((premises, target))
        
        # Update State
        known_facts.add(target)
        used_vars.add(target)
        ground_truth_sequence.append(target)

    # 3. Add Distractors
    # Distractors must NOT fire.
    # Easiest way: Ensure at least one premise is NOT in the final 'known_facts'.
    # OR: The target is already known (redundant/cyclic).
    # But to be safe and clean, we use variables that are never true.
    
    distractor_rules = []
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 200:
        attempts += 1
        
        # Pick random variables
        s_count = 2 if random.random() < config.branch_prob else 1
        s = random.sample(available_vars, s_count)
        t = random.choice(available_vars)
        
        # Rule Validation:
        # 1. Don't replicate an existing rule
        if any((set(s) == set(tr[0]) and t == tr[1]) for tr in true_rules + distractor_rules):
            continue
            
        # 2. Critical: Ensure this rule doesn't accidentally trigger using valid facts
        # If all 's' are in the final 'known_facts', this rule would fire!
        # We want strict control. So, valid distractors usually involve an 'Unknown' variable.
        # But for difficulty, let's allow "Dead Ends":
        # Rules that fire but lead to a variable we don't care about?
        # No, to keep it simple: Distractors should FAIL to fire.
        
        # Check if premises are a subset of the TRUE facts
        if set(s).issubset(known_facts):
            # If it fires, does it mess up our order? 
            # It creates a parallel true path. Let's avoid this for now to keep Target deterministic.
            continue
            
        distractor_rules.append((s, t))

    # 4. Format Output
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    # Stringify rules: "A&B>C" or "A>B"
    rule_strs = []
    for premises, target in all_rules:
        lhs = "&".join(premises)
        rule_strs.append(f"{lhs}>{target}")
        
    # Input: "Facts: A B | Rules: A&B>C D>E ..."
    input_str = f"Facts: {' '.join(start_facts_list)} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(ground_truth_sequence)
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    # Space splitting
    raw_parts = text.split(' ')
    
    for part in raw_parts:
        if not part: continue # Skip empty strings
        
        # Check if exact match
        if part in VOCAB:
            tokens.append(VOCAB[part])
            continue
            
        # Parse composite "A&B>C"
        # We allow A, B, C, &, >, |
        buffer = ""
        for char in part:
            if char in ['&', '>', '|']:
                if buffer and buffer in VOCAB: 
                    tokens.append(VOCAB[buffer])
                tokens.append(VOCAB[char])
                buffer = ""
            else:
                buffer += char
        
        # Flush buffer
        if buffer and buffer in VOCAB: 
            tokens.append(VOCAB[buffer])

    if len(tokens) > seq_len - 1:
        return None
        
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens

# --- Standard Dataset Boilerplate (Same as before) ---
def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    # FIXED: Do not reset seed here!
    
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
            
            if (input_tokens is None) or (label_tokens is None):
                continue
                
            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))
            
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(valid_count) # Using Unique ID (0-based)
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
    # Set global seed once
    random.seed(config.seed)
    np.random.seed(config.seed)

    print(f"Generating Branching Logic in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()