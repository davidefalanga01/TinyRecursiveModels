import os
import json
import string
import random
import numpy as np
import networkx as nx
from typing import Optional, List, Tuple
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# --- Vocabulary Definition ---
# We use A-Z for variables.
# We define specific tokens for structure to make it easier for the Tiny Model.
VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '|', 'end', 'pad']

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
    
    # Task Specific Configs
    num_vars: int = 26       # Size of variable pool (A-Z)
    min_steps: int = 2       # Min length of chain (A->B->C is 2 steps)
    max_steps: int = 6       # Max length of chain
    num_distractors: int = 3 # Number of fake rules to add
    seed: int = 42

def generate_chain_sample(config: DataProcessConfig) -> Tuple[str, str]:
    """
    Generates a linear logic chain with distractors.
    Returns: (input_string, target_string)
    """
    # 1. Select variables for the valid chain
    chain_len = random.randint(config.min_steps, config.max_steps)
    available_vars = VARS[:config.num_vars]
    
    # Ensure we have enough vars
    if len(available_vars) < chain_len + 1:
        available_vars = VARS # Fallback to all if pool is too small
        
    chain_vars = random.sample(available_vars, chain_len + 1)
    
    # Define the ground truth path: A -> B -> C
    true_rules = []
    for i in range(len(chain_vars) - 1):
        source = chain_vars[i]
        target = chain_vars[i+1]
        true_rules.append((source, target))
    
    start_fact = chain_vars[0]
    ground_truth_sequence = chain_vars[1:] # The solution path (B, C...)

    # 2. Add Distractors
    # We must ensure distractors do not create cycles or shortcuts
    distractor_rules = []
    existing_edges = set(true_rules)
    
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 100:
        attempts += 1
        s, t = random.sample(available_vars, 2)
        
        # Validation
        if s == t: continue
        if (s, t) in existing_edges: continue
        
        # Check for cycles/shortcuts using a temp graph
        temp_edges = true_rules + distractor_rules + [(s, t)]
        G = nx.DiGraph(temp_edges)
        
        if not nx.is_directed_acyclic_graph(G):
            continue
            
        # Ensure we didn't accidentally create a shortcut from Start to End
        # BFS from start_fact should still produce the exact same path length to the end
        try:
            path = nx.shortest_path(G, source=start_fact, target=chain_vars[-1])
            if len(path) != len(chain_vars):
                continue # A shortcut was created
        except nx.NetworkXNoPath:
            pass # This is fine, it means the distractor is disconnected or downstream
            
        distractor_rules.append((s, t))
        existing_edges.add((s, t))

    # 3. Format Strings
    all_rules = true_rules + distractor_rules
    random.shuffle(all_rules) # Shuffle to remove positional clues
    
    # Format: "A>B"
    rule_strs = [f"{r[0]}>{r[1]}" for r in all_rules]
    
    # Input: "Facts: A | Rules: C>D A>B"
    input_str = f"Facts: {start_fact} | Rules: {' '.join(rule_strs)} | Target:"
    
    # Output: "B C" (The sequence of deduction)
    target_str = " ".join(ground_truth_sequence)
    
    return input_str, target_str

def tokenize(text: str, seq_len: int = 64) -> Optional[List[int]]:
    """
    Tokenizes the string based on space separation and custom logic.
    """
    tokens = []
    # Split by space to handle keywords like "Facts:" cleanly
    raw_tokens = text.split(' ')
    
    for rt in raw_tokens:
        if rt in VOCAB:
            tokens.append(VOCAB[rt])
        else:
            # Handle composite tokens like "A>B" or "Target:" if missed
            # We assume inputs are clean, but let's be robust
            # If "A>B", split into A, >, B
            if '>' in rt and len(rt) > 1:
                parts = rt.partition('>')
                for p in parts:
                    if p in VOCAB:
                        tokens.append(VOCAB[p])
            else:
                # Fallback (should not happen with this generator)
                pass

    if len(tokens) > seq_len - 1:
        return None
        
    # Pad
    tokens.append(VOCAB['end'])
    padding = [VOCAB['pad']] * (seq_len - len(tokens))
    tokens = tokens + padding
    return tokens

# Dataset generation
def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    np.random.seed(config.seed)
    random.seed(config.seed)

    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0
    
    # Initial indices
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    # Counters for valid samples
    valid_count = 0
    
    with tqdm(total=num_samples, desc=f"Generating {set_name}") as pbar:
        while valid_count < num_samples:
            inp_str, targ_str = generate_chain_sample(config)
            
            # Create full sequence for training: Input + Target
            # But the model inputs should just be Input, labels should be Input+Target
            # TRM usually expects:
            # Input:  [Tokens....]
            # Label:  [Tokens....] (Shifted or masked)
            
            # Based on your previous snippet, you want:
            # Input: "Problem |-"
            # Label: "Problem |- Answer"
            
            # Adapting to our format:
            full_text = f"{inp_str} {targ_str}"
            
            # Tokenize
            # For TRM 'input' usually masks the answer or provides the prompt
            input_tokens = tokenize(inp_str, config.seq_len)
            label_tokens = tokenize(full_text, config.seq_len)
            
            if (input_tokens is None) or (label_tokens is None):
                continue 
            
            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))
            
            example_id += 1
            puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            results["group_indices"].append(puzzle_id)
            
            valid_count += 1
            pbar.update(1)

    # Convert to numpy arrays
    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = {
        "seq_len": config.seq_len,
        "vocab_size": len(VOCAB),
        "pad_id": VOCAB['pad'],
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": 1,
        "total_groups": len(final_results["group_indices"]) - 1,
        "mean_puzzle_examples": 1.0, # 1 example per puzzle here
        "total_puzzles": puzzle_id,
        "sets": ["all"]
    }

    # Save dataset
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Identifiers file
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
        
    # Save Vocab for later use
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(VOCAB, f, indent=4)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    print(f"Generating Linear Chain Logic Dataset in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()