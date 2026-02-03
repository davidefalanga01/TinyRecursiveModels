import os
import json
import string
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser
from typing import Optional, List, Tuple

# --- FIX: REMOVED 'end' and 'pad' from SPECIALS ---
# They are already defined manually as 0 and 1.
VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '|'] 

VOCAB = {
    'pad': 0,
    'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_chain"
    seq_len: int = 64
    num_train: int = 50000
    num_test: int = 5000
    num_vars: int = 26
    min_steps: int = 2
    max_steps: int = 6
    num_distractors: int = 3
    seed: int = 42

def generate_chain_sample(config: DataProcessConfig) -> Tuple[str, str]:
    chain_len = random.randint(config.min_steps, config.max_steps)
    available_vars = VARS[:config.num_vars]
    if len(available_vars) < chain_len + 1: available_vars = VARS
        
    chain_vars = random.sample(available_vars, chain_len + 1)
    true_rules = [(chain_vars[i], chain_vars[i+1]) for i in range(len(chain_vars) - 1)]
    start_fact = chain_vars[0]
    ground_truth = chain_vars[1:]

    # Distractors
    distractor_rules = []
    existing_edges = set(true_rules)
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 100:
        attempts += 1
        s, t = random.sample(available_vars, 2)
        if s == t or (s, t) in existing_edges: continue
        
        temp_edges = true_rules + distractor_rules + [(s, t)]
        G = nx.DiGraph(temp_edges)
        if not nx.is_directed_acyclic_graph(G): continue
        
        try:
            if len(nx.shortest_path(G, start_fact, chain_vars[-1])) != len(chain_vars): continue
        except: pass
            
        distractor_rules.append((s, t))
        existing_edges.add((s, t))

    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules)
    
    rule_strs = [f"{r[0]}>{r[1]}" for r in all_rules]
    input_str = f"Facts: {start_fact} | Rules: {' '.join(rule_strs)} | Target:"
    target_str = " ".join(ground_truth)
    return input_str, target_str

def tokenize(text: str, seq_len: int = 64) -> Optional[List[int]]:
    tokens = []
    for rt in text.split(' '):
        if rt in VOCAB:
            tokens.append(VOCAB[rt])
        else:
            if '>' in rt and len(rt) > 1:
                parts = rt.partition('>')
                for p in parts:
                    if p in VOCAB: tokens.append(VOCAB[p])
    
    if len(tokens) > seq_len - 1: return None
    
    tokens.append(VOCAB['end'])
    tokens += [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens

def convert_subset(set_name, config, num):
    # Fix: Do NOT reset seed here with the same value for every subset
    # np.random.seed(config.seed); random.seed(config.seed) 
    
    # Initialize results dictionary
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    
    # Track sequence and puzzle counts
    e_id = 0 # example id
    p_id = 0 # puzzle id
    
    # Start indices (required by the original paper's data loader)
    results["puzzle_indices"].append(0) 
    results["group_indices"].append(0)
    
    valid = 0
    with tqdm(total=num, desc=f"Generating {set_name}") as pbar:
        while valid < num:
            inp, targ = generate_chain_sample(config)
            full = f"{inp} {targ}"
            it = tokenize(inp, config.seq_len)
            ft = tokenize(full, config.seq_len) # Full tokenized sequence
            
            if it and ft:
                results["inputs"].append(np.array(it))
                results["labels"].append(np.array(ft))
                
                # Update counters
                e_id += 1 
                p_id += 1
                
                # Append indices for this batch
                results["puzzle_indices"].append(e_id)
                results["group_indices"].append(p_id)
                
                # FIXED: Use a single shared ID (0) for all logic samples
                # This treats the logic task as a single "puzzle" type to learn 
                results["puzzle_identifiers"].append(0) 
                
                valid += 1
                pbar.update(1)

    # Convert to numpy arrays
    final_inputs = np.stack(results["inputs"])
    final_labels = np.stack(results["labels"])
    final_puzzle_identifiers = np.array(results["puzzle_identifiers"], dtype=np.int32)
    final_puzzle_indices = np.array(results["puzzle_indices"], dtype=np.int32)
    final_group_indices = np.array(results["group_indices"], dtype=np.int32)
    
    # Save the files
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "all__inputs.npy"), final_inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), final_labels)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), final_puzzle_identifiers)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), final_puzzle_indices)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), final_group_indices)
    
    # Update dataset.json with the CORRECT number of puzzle identifiers
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump({
            "seq_len": config.seq_len,
            "vocab_size": len(VOCAB),
            "pad_id": 0,
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "num_puzzle_identifiers": 1, # FIXED: Only 1 type of "puzzle" (logic chain)
            "total_groups": p_id,
            "mean_puzzle_examples": 1.0,
            "total_puzzles": p_id,
            "sets": ["all"]
        }, f)
    
    # Save general vocab and identifiers (needed once per dataset)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f: 
        json.dump(VOCAB, f)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f: 
        # Create a list of identifiers matching the 'valid' count
        json.dump(["logic_task"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    # Set global seed once
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Generating FIXED Logic Chain in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()