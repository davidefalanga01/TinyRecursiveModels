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
# Added '~' for negation, '|' for OR.
# Structure: Facts, Rules, Target
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '~', '|']

VOCAB = {
    'pad': 0,
    'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_or"
    seq_len: int = 160 # Increased slightly for more complex rules
    num_train: int = 50000
    num_test: int = 5000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 6
    negation_prob: float = 0.3  # Chance a rule has a (~NOT) condition
    or_prob: float = 0.3        # Chance a premise clause is an (A|B) disjunction
    num_distractors: int = 4
    seed: int = 42

def get_stratified_target(start_facts: Set[str], rules: List[Tuple[List[List[str]], List[str], str]]) -> List[str]:
    """
    Simulates logic flow with Negation and Disjunction (OR).
    Rule format: (positive_clauses, negative_premises, target)
    positive_clauses: List of Lists. [[A, B], [C]] -> (A OR B) AND C
    """
    known = {f: 0 for f in start_facts} # Fact -> Depth
    derived = []
    
    changed = True
    while changed:
        changed = False
        for pos_clauses, neg_prem, target in rules:
            if target in known: continue
            
            # --- Condition 1: Negation (CWA) ---
            # All Negative Premises must be UNKNOWN
            if any(n in known for n in neg_prem):
                continue
                
            # --- Condition 2: Positive Disjunctions (OR/AND) ---
            # Every clause must be satisfied. 
            # A clause is satisfied if AT LEAST ONE var is in known.
            
            clause_depths = []
            all_clauses_satisfied = True
            
            for clause in pos_clauses:
                # Find all known vars in this clause
                known_vars_in_clause = [v for v in clause if v in known]
                
                if not known_vars_in_clause:
                    all_clauses_satisfied = False
                    break
                
                # The depth of this clause is determined by the shallowest available fact.
                # Logic: If A (depth 1) and B (depth 5) are both true, A|B is satisfied at step 1.
                clause_depths.append(min(known[v] for v in known_vars_in_clause))
            
            if not all_clauses_satisfied:
                continue

            # If fires:
            # Depth is max(clause_depths) + 1. 
            new_depth = max(clause_depths) + 1 if clause_depths else 1
            
            known[target] = new_depth
            derived.append((new_depth, target))
            changed = True
    
    # Sort: Depth ASC, Name ASC
    derived.sort(key=lambda x: (x[0], x[1]))
    return [x[1] for x in derived]

def generate_negation_or_sample(config: DataProcessConfig) -> Tuple[str, str]:
    available_vars = VARS[:config.num_vars]
    
    # 1. Partition Variables
    random.shuffle(available_vars)
    split_idx = random.randint(len(available_vars)//2, len(available_vars)-2)
    
    allowed_true_vars = set(available_vars[:split_idx])
    forbidden_vars = set(available_vars[split_idx:]) 
    
    # 2. Initialize Facts
    num_start = random.randint(1, 2)
    start_facts = set(random.sample(list(allowed_true_vars), num_start))
    start_facts_list = sorted(list(start_facts))
    
    known_facts = set(start_facts)
    true_rules = [] # List of (pos_clauses, neg_prem, target)
    
    # 3. Build Derivation Chain
    num_layers = random.randint(config.min_layers, config.max_layers)
    used_vars = set(known_facts)
    
    for _ in range(num_layers):
        potential_targets = [v for v in allowed_true_vars if v not in used_vars]
        if not potential_targets: break
            
        target = random.choice(potential_targets)
        
        # --- Build Positive Premises (with OR) ---
        # We need 1 or 2 "clauses" (AND components)
        # Each clause can be a single var "A" or a disjunction "A|B"
        
        num_clauses = random.randint(1, 2)
        pos_clauses = []
        
        for _ in range(num_clauses):
            # To ensure the rule fires, we MUST pick at least one known fact
            trigger = random.choice(list(known_facts))
            
            if random.random() < config.or_prob:
                # Create a Disjunction: (Trigger | Distractor)
                # The distractor can be anything: True, False (Forbidden), or Unknown (Future)
                # This teaches the model: True | False -> True
                distractor_opt = random.choice(available_vars)
                clause = [trigger, distractor_opt]
                random.shuffle(clause)
                pos_clauses.append(clause)
            else:
                # Standard single atom
                pos_clauses.append([trigger])
        
        # --- Build Negative Premises ---
        neg_prem = []
        if random.random() < config.negation_prob:
            neg_prem.append(random.choice(list(forbidden_vars)))
            
        true_rules.append((pos_clauses, neg_prem, target))
        
        known_facts.add(target)
        used_vars.add(target)

    # 4. Generate Distractors
    distractor_rules = []
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 200:
        attempts += 1
        
        # We want "Tricky" invalid rules.
        
        # Type A: Blocked by Negation 
        # (A|B) & ~C > D. But C is True.
        if len(known_facts) >= 2:
            # Pick a valid positive clause
            p = random.choice(list(known_facts))
            
            # Pick a blocker that is actually true
            n = random.choice(list(known_facts)) 
            if p == n: continue
            
            t = random.choice(available_vars)
            if t in known_facts: continue 
            
            # This rule implies: P | X & ~N > T
            # We add a random 'X' to the OR to make it look robust, but ~N kills it.
            filler = random.choice(available_vars)
            clause = [p, filler]
            random.shuffle(clause)
            
            distractor_rules.append(([clause], [n], t))
            continue
            
        # Type B: Failed OR Condition
        # A | B > C. Both A and B are False (Forbidden).
        if len(forbidden_vars) >= 2:
            bad1 = random.choice(list(forbidden_vars))
            bad2 = random.choice(list(forbidden_vars))
            
            t = random.choice(available_vars)
            if t in known_facts: continue
            
            distractor_rules.append(([[bad1, bad2]], [], t))

    # 5. Format Output
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for pos_clauses, neg, target in all_rules:
        # Format: "A|B & C & ~D > E"
        
        # 1. Format Positive Clauses (A|B)
        clause_strs = []
        for clause in pos_clauses:
            # Unique and Sort
            u_clause = sorted(list(set(clause)))
            clause_strs.append("|".join(u_clause))
        
        # 2. Format Negative Premises (~D)
        neg_strs = [f"~{n}" for n in sorted(neg)]
        
        # 3. Join with &
        lhs_parts = sorted(clause_strs) + neg_strs
        lhs = "&".join(lhs_parts)
        
        rule_strs.append(f"{lhs}>{target}")
        
    # Calculate Stratified Target (Ground Truth)
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
        i = 0
        while i < len(part):
            char = part[i]
            
            # Check for operators: &, >, |
            if char in ['&', '>', '|']:
                if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])
                tokens.append(VOCAB[char])
                buffer = ""
                i += 1
            # Check for Negation Tilde '~'
            elif char == '~':
                if buffer and buffer in VOCAB: tokens.append(VOCAB[buffer])
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
            inp_str, targ_str = generate_negation_or_sample(config)
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
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "total_groups": len(final_results["group_indices"]) - 1,
            "mean_puzzle_examples": 1.0,
            "num_puzzle_identifiers": valid_count,
            "total_puzzles": valid_count,
            "sets": ["all"]
        }, f)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([f"logic_{i}" for i in range(valid_count)], f)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(VOCAB, f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed); np.random.seed(config.seed)
    print(f"Generating Logic (OR/AND/NOT) in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()