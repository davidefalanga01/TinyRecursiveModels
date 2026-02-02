import random
import networkx as nx
import string
import numpy as np
import os
import json
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser
from typing import List, Optional, Tuple

# Vocabulary now includes '~'
VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '~', '|', 'end', 'pad']

VOCAB = {
    'pad': 0, 'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_negation"
    seq_len: int = 128
    num_train: int = 30000
    num_test: int = 3000
    num_vars: int = 26
    min_layers: int = 2
    max_layers: int = 5
    negation_prob: float = 0.3 # 30% of rules will have a "NOT" clause
    seed: int = 42

def generate_negation_sample(config: DataProcessConfig):
    available_vars = VARS[:config.num_vars]
    
    # 1. Start Facts
    num_start = random.randint(2, 4)
    start_facts = set(random.sample(available_vars, num_start))
    
    # We maintain the "Current Truth" state
    current_facts = set(start_facts)
    true_rules = []
    derivation_sequence = []
    
    # 2. Build Layer by Layer
    # We must ensure we don't create contradictions or cycles.
    used_targets = set(start_facts)
    
    for _ in range(config.max_layers):
        potential_targets = [v for v in available_vars if v not in used_targets]
        if not potential_targets: break
            
        target = random.choice(potential_targets)
        
        # Create a rule: (Positive_Premises, Negative_Premises) > Target
        # To ensure it FIRES, we pick Positive from current_facts, Negative from OUTSIDE current_facts
        
        # Positive Premise (Must exist)
        pos_premise = random.choice(list(current_facts))
        
        # Negative Premise (Optional)
        neg_premise = None
        if random.random() < config.negation_prob:
            # Pick a variable that is currently FALSE
            # And to make it hard, pick one that MIGHT appear in distractors later?
            # For now, just pick any False variable.
            false_vars = [v for v in available_vars if v not in current_facts]
            if false_vars:
                neg_premise = random.choice(false_vars)
        
        # Store Rule
        true_rules.append(((pos_premise, neg_premise), target))
        
        # Update State
        current_facts.add(target)
        used_targets.add(target)
        derivation_sequence.append(target)

    # 3. Add "Trap" Rules (Distractors)
    # These are rules that look valid but are blocked by a Negation
    # e.g., A & ~B > Z. (Where A is True, but B is ALSO True, so Z doesn't fire)
    distractor_rules = []
    attempts = 0
    while len(distractor_rules) < 4 and attempts < 200:
        attempts += 1
        
        # Try to create a blocked rule
        # Pick a known fact as Positive
        if not list(current_facts): continue
        p = random.choice(list(current_facts))
        
        # Pick a KNOWN fact as Negative (Blocker)
        n = random.choice(list(current_facts))
        if p == n: continue
        
        t = random.choice(available_vars)
        if t in current_facts: continue # Don't target something that's already true
        
        # This rule (P & ~N > T) will fail because N is True!
        distractor_rules.append(((p, n), t))

    # 4. Format
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = []
    for (pos, neg), target in all_rules:
        if neg:
            rule_strs.append(f"{pos}&~{neg}>{target}")
        else:
            rule_strs.append(f"{pos}>{target}")
            
    input_str = f"Facts: {' '.join(sorted(list(start_facts)))} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(derivation_sequence)
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    raw_parts = text.split(' ')
    for part in raw_parts:
        if part in VOCAB:
            tokens.append(VOCAB[part])
            continue
        buffer = ""
        for char in part:
            if char in ['&', '>', '~']:
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

# --- Boilerplate ---
def convert_subset(set_name, config, num):
    # Same boilerplate as previous scripts...
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id, example_id = 0, 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    with tqdm(total=num) as pbar:
        valid = 0
        while valid < num:
            inp, targ = generate_negation_sample(config)
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
                
    # Save (same as before)...
    final_results = {k: np.array(v) if k != "puzzle_identifiers" else np.array(v, dtype=np.int32) for k, v in results.items()}
    # Fix dict-list mismatch in simple code block above for brevity
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
    print(f"Generating Negation Logic in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()