import os
import json
import string
import random
import numpy as np
import collections
from typing import Optional, List, Tuple, Set
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# --- Vocabulary ---
VARS = list(string.ascii_uppercase)
# Added '~' for negation. Structure: Facts, Rules, Target
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '~', '|']

VOCAB = {
    'pad': 0,
    'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_negation"
    seq_len: int = 128
    num_train: int = 50000
    num_test: int = 5000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 6
    negation_prob: float = 0.4  # 40% chance a rule has a (~NOT) condition
    num_distractors: int = 4
    seed: int = 42

def get_stratified_target(start_facts: Set[str], rules: List[Tuple[List[str], List[str], str]]) -> List[str]:
    """
    Simulates logic flow with Negation.
    Rule format: (positive_premises, negative_premises, target)
    """
    known = {f: 0 for f in start_facts} # Fact -> Depth
    derived = []
    
    changed = True
    while changed:
        changed = False
        for pos_prem, neg_prem, target in rules:
            if target in known: continue
            
            # Condition 1: All Positive Premises must be KNOWN
            if not all(p in known for p in pos_prem):
                continue
                
            # Condition 2: All Negative Premises must be UNKNOWN (Closed World Assumption)
            # In our strict generation, these are "Forbidden Vars" that will never be true.
            if any(n in known for n in neg_prem):
                continue

            # If fires:
            # Depth is max(pos_depths) + 1. Negative premises don't add depth (they are static checks).
            # If pos_prem is empty (rare base fact rule), depth is 1.
            current_depths = [known[p] for p in pos_prem]
            new_depth = max(current_depths) + 1 if current_depths else 1
            
            known[target] = new_depth
            derived.append((new_depth, target))
            changed = True
    
    # Sort: Depth ASC, Name ASC
    derived.sort(key=lambda x: (x[0], x[1]))
    return [x[1] for x in derived]

def generate_negation_sample(config: DataProcessConfig) -> Tuple[str, str]:
    available_vars = VARS[:config.num_vars]
    
    # 1. Partition Variables to ensure consistency
    # We decide UP FRONT which variables are allowed to be True, and which are Forbidden.
    # This prevents the "Time Travel Paradox" where we negated B early on, but B becomes true later.
    random.shuffle(available_vars)
    split_idx = random.randint(len(available_vars)//2, len(available_vars)-2)
    
    allowed_true_vars = set(available_vars[:split_idx])
    forbidden_vars = set(available_vars[split_idx:]) # These can safely be used in ~NOT clauses
    
    # 2. Initialize Facts
    num_start = random.randint(2, 4)
    # Start facts must come from allowed_true_vars
    start_facts = set(random.sample(list(allowed_true_vars), num_start))
    start_facts_list = sorted(list(start_facts))
    
    known_facts = set(start_facts)
    true_rules = [] # List of ([pos], [neg], target)
    
    # 3. Build Derivation Chain
    num_layers = random.randint(config.min_layers, config.max_layers)
    used_vars = set(known_facts)
    
    for _ in range(num_layers):
        # Pick a target that is Allowed but not yet Known
        potential_targets = [v for v in allowed_true_vars if v not in used_vars]
        if not potential_targets: break
            
        target = random.choice(potential_targets)
        
        # Build Premises
        # We need at least 1 positive premise from Known Facts to anchor the depth
        pos_prem = random.sample(list(known_facts), random.randint(1, 2))
        
        neg_prem = []
        # Chance to add a Negative Premise
        if random.random() < config.negation_prob:
            # CRITICAL: Pick negative premise from FORBIDDEN vars
            # This guarantees the rule is valid and won't be invalidated later.
            neg_prem.append(random.choice(list(forbidden_vars)))
            
        true_rules.append((pos_prem, neg_prem, target))
        
        known_facts.add(target)
        used_vars.add(target)

    # 4. Generate Distractors (The Learning Signal)
    distractor_rules = []
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 200:
        attempts += 1
        
        # We want to generate "Tricky" invalid rules.
        
        # Type A: Blocked by Negation (The most important type for this task)
        # Rule: A & ~B > C. 
        # Make A True, but make B True as well. Rule fails.
        if len(known_facts) >= 2:
            p = random.choice(list(known_facts))
            n = random.choice(list(known_facts)) # N is True!
            if p == n: continue
            
            t = random.choice(available_vars)
            if t in known_facts: continue 
            
            # This rule looks like P & ~N > T. 
            # It fails because N is actually in facts.
            distractor_rules.append(([p], [n], t))
            continue
            
        # Type B: Standard False Premise
        # Rule: A & B > C. A is True, B is False (Forbidden).
        p = random.choice(list(known_facts))
        bad = random.choice(list(forbidden_vars))
        t = random.choice(available_vars)
        if t in known_facts: continue
        
        distractor_rules.append(([p, bad], [], t))

    # 5. Format Output
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for pos, neg, target in all_rules:
        # Format: "A&B&~C>D"
        # Sort premises for stability
        parts = sorted(pos) + [f"~{n}" for n in sorted(neg)]
        lhs = "&".join(parts)
        rule_strs.append(f"{lhs}>{target}")
        
    # Calculate Stratified Target (Ground Truth)
    # We recalculate strictly from start facts and rules to ensure perfect consistency
    final_target = get_stratified_target(set(start_facts_list), true_rules)
    
    input_str = f"Facts: {' '.join(start_facts_list)} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(final_target)
    
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
        # Improved parsing loop to handle '~' correctly
        i = 0
        while i < len(part):
            char = part[i]
            
            # Check for operators
            if char in ['&', '>', '|']:
                if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])
                tokens.append(VOCAB[char])
                buffer = ""
                i += 1
            # Check for Negation Tilde '~'
            elif char == '~':
                if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])
                # We treat '~' as a standalone token before the var
                tokens.append(VOCAB['~'])
                buffer = ""
                i += 1
            else:
                buffer += char
                i += 1
                
        if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])

    if len(tokens) > seq_len - 1: return None
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens

# --- Standard Boilerplate ---
def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0); results["group_indices"].append(0)
    
    valid_count = 0
    with tqdm(total=num_samples, desc=f"Generating {set_name}") as pbar:
        while valid_count < num_samples:
            inp_str, targ_str = generate_negation_sample(config)
            full_text = f"{inp_str} {targ_str}"
            
            input_tokens = tokenize(inp_str, config.seq_len)
            label_tokens = tokenize(full_text, config.seq_len)
            
            if (input_tokens is None) or (label_tokens is None): continue
                
            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))
            example_id += 1; puzzle_id += 1
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
    
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    for k, v in final_results.items(): np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump({
            "seq_len": config.seq_len,
            "vocab_size": len(VOCAB),
            "pad_id": VOCAB['pad'],
            "num_puzzle_identifiers": valid_count,
            "sets": ["all"]
        }, f)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([f"logic_{i}" for i in range(valid_count)], f)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(VOCAB, f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed); np.random.seed(config.seed)
    print(f"Generating Negation Logic in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()