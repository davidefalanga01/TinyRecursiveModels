import os
import json
import string
import random
import numpy as np
from typing import Optional, List, Tuple
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# --- Vocabulary: All Operators Included ---
VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '+', '~', '|', 'end', 'pad']

VOCAB = {
    'pad': 0, 'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_mixed"
    seq_len: int = 160          # Increased for complexity
    num_train: int = 50000
    num_test: int = 4000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 6
    
    # Probabilities for Rule Types (Must sum to <= 1.0)
    # Remaining probability is for Simple Rules (A > B)
    prob_and: float = 0.25      # A & B > C
    prob_or: float = 0.25       # A + B > C
    prob_not: float = 0.25      # A & ~B > C
    
    seed: int = 42

def generate_mixed_sample(config: DataProcessConfig) -> Tuple[str, str]:
    available_vars = VARS[:config.num_vars]
    
    # 1. Start Facts
    num_start = random.randint(2, 4)
    start_facts = set(random.sample(available_vars, num_start))
    
    current_facts = set(start_facts)
    true_rules = []
    derivation_sequence = []
    used_targets = set(start_facts)
    
    # 2. Build Valid Derivation Path
    for _ in range(random.randint(config.min_layers, config.max_layers)):
        potential_targets = [v for v in available_vars if v not in used_targets]
        if not potential_targets: break
        
        target = random.choice(potential_targets)
        
        # Decide Rule Type
        r = random.random()
        
        # --- TYPE 1: AND (A & B > C) ---
        if r < config.prob_and:
            if len(current_facts) < 2: continue # fallback to simple
            p1, p2 = random.sample(list(current_facts), 2)
            true_rules.append(({'type': 'AND', 'p': [p1, p2], 't': target}))
            
        # --- TYPE 2: OR (A + B > C) ---
        # Fires if at least ONE premise is true.
        elif r < (config.prob_and + config.prob_or):
            # Pick one valid premise
            p1 = random.choice(list(current_facts))
            # Pick a second premise (can be T or F, doesn't matter)
            p2 = random.choice(available_vars)
            if p2 == p1: p2 = random.choice([v for v in available_vars if v != p1])
            
            # Randomize order so the model doesn't learn "First is always true"
            ps = [p1, p2]
            random.shuffle(ps)
            true_rules.append(({'type': 'OR', 'p': ps, 't': target}))
            
        # --- TYPE 3: NOT / Inhibition (A & ~B > C) ---
        # Fires if A is True AND B is False
        elif r < (config.prob_and + config.prob_or + config.prob_not):
            p1 = random.choice(list(current_facts))
            # Find a variable that is currently FALSE
            false_vars = [v for v in available_vars if v not in current_facts]
            if not false_vars: continue
            p2 = random.choice(false_vars)
            
            true_rules.append(({'type': 'NOT', 'p': [p1, p2], 't': target}))
            
        # --- TYPE 4: Simple (A > C) ---
        else:
            p1 = random.choice(list(current_facts))
            true_rules.append(({'type': 'SIMPLE', 'p': [p1], 't': target}))

        # Update State
        current_facts.add(target)
        used_targets.add(target)
        derivation_sequence.append(target)

    # 3. Add Distractors (Rules that FAIL)
    # We need distractors for every type to ensure the model isn't guessing.
    distractor_rules = []
    attempts = 0
    
    # We want roughly 5 distractors
    while len(distractor_rules) < 5 and attempts < 200:
        attempts += 1
        t = random.choice(available_vars)
        if t in current_facts: continue # Don't target known facts
        
        r_type = random.choice(['AND', 'OR', 'NOT', 'SIMPLE'])
        
        if r_type == 'SIMPLE':
            # Fail: Premise is False
            false_vars = [v for v in available_vars if v not in current_facts]
            if not false_vars: continue
            p = random.choice(false_vars)
            distractor_rules.append({'type': 'SIMPLE', 'p': [p], 't': t})
            
        elif r_type == 'AND':
            # Fail: At least one premise is False
            # Option A: T & F
            # Option B: F & F
            p1 = random.choice(available_vars) # Random T or F
            false_vars = [v for v in available_vars if v not in current_facts]
            if not false_vars: continue
            p2 = random.choice(false_vars) # Definitely F
            ps = [p1, p2]; random.shuffle(ps)
            distractor_rules.append({'type': 'AND', 'p': ps, 't': t})
            
        elif r_type == 'OR':
            # Fail: BOTH premises must be False
            false_vars = [v for v in available_vars if v not in current_facts]
            if len(false_vars) < 2: continue
            p1, p2 = random.sample(false_vars, 2)
            distractor_rules.append({'type': 'OR', 'p': [p1, p2], 't': t})
            
        elif r_type == 'NOT':
            # Fail: A & ~B > C
            # Fail Case 1: A is False (irrelevant B)
            # Fail Case 2: A is True, but B is ALSO True (Blocked!) <--- This is the important trap
            if len(current_facts) >= 2:
                # Create a Trap
                p1 = random.choice(list(current_facts)) # A is True
                p2 = random.choice(list(current_facts)) # B is True (Blocker!)
                if p1 != p2:
                    distractor_rules.append({'type': 'NOT', 'p': [p1, p2], 't': t})

    # 4. Format Output
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for rule in all_rules:
        if rule['type'] == 'SIMPLE':
            rule_strs.append(f"{rule['p'][0]}>{rule['t']}")
        elif rule['type'] == 'AND':
            rule_strs.append(f"{rule['p'][0]}&{rule['p'][1]}>{rule['t']}")
        elif rule['type'] == 'OR':
            rule_strs.append(f"{rule['p'][0]}+{rule['p'][1]}>{rule['t']}")
        elif rule['type'] == 'NOT':
            # p[0] is positive, p[1] is negative
            rule_strs.append(f"{rule['p'][0]}&~{rule['p'][1]}>{rule['t']}")
            
    input_str = f"Facts: {' '.join(sorted(list(start_facts)))} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(derivation_sequence)
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    # Split strictly by space first
    parts = text.split(' ')
    for part in parts:
        if part in VOCAB:
            tokens.append(VOCAB[part])
        else:
            # Parse complex tokens like "A&~B>C" or "A+B>C"
            buffer = ""
            for char in part:
                if char in ['&', '+', '~', '>', '|']:
                    if buffer in VOCAB: tokens.append(VOCAB[buffer])
                    tokens.append(VOCAB[char])
                    buffer = ""
                else:
                    buffer += char
            if buffer in VOCAB: tokens.append(VOCAB[buffer])
            
    if len(tokens) > seq_len - 1: return None
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens

# --- Standard Boilerplate ---
def convert_subset(set_name, config, num):
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id, example_id = 0, 0
    results["puzzle_indices"].append(0); results["group_indices"].append(0)
    
    with tqdm(total=num) as pbar:
        valid = 0
        while valid < num:
            inp, targ = generate_mixed_sample(config)
            full = f"{inp} {targ}"
            
            in_tok = tokenize(inp, config.seq_len)
            out_tok = tokenize(full, config.seq_len)
            
            if in_tok and out_tok:
                results["inputs"].append(np.array(in_tok))
                results["labels"].append(np.array(out_tok))
                example_id += 1; puzzle_id += 1
                results["puzzle_indices"].append(example_id)
                results["group_indices"].append(puzzle_id)
                results["puzzle_identifiers"].append(0)
                valid += 1
                pbar.update(1)

    final_results = {k: np.array(v) if k != "puzzle_identifiers" else np.array(v, dtype=np.int32) for k, v in results.items()}
    final_results["puzzle_identifiers"] = np.array(results["puzzle_identifiers"], dtype=np.int32)
    final_results["inputs"] = np.stack(results["inputs"])
    final_results["labels"] = np.stack(results["labels"])
    final_results["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
    final_results["group_indices"] = np.array(results["group_indices"], dtype=np.int32)

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    for k,v in final_results.items(): np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    meta = {"seq_len": config.seq_len, "vocab_size": len(VOCAB), "pad_id": 0, "sets": ["all"]}
    with open(os.path.join(save_dir, "dataset.json"), "w") as f: json.dump(meta, f)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f: json.dump(VOCAB, f)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f: json.dump(["<blank>"], f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    print(f"Generating MIXED Logic (AND, OR, NOT) in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()