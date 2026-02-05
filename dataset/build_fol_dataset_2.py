import os
import json
import string
import random
import argparse
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Tuple, Set
from dataclasses import dataclass

# --- Configuration ---
@dataclass
class DataProcessConfig:
    output_dir: str = "data/logic_chain"
    seq_len: int = 256
    num_train: int = 20000
    num_test: int = 2000
    num_vars: int = 26
    min_steps: int = 2
    max_steps: int = 6
    num_distractors: int = 4
    multi_start: bool = True
    seed: int = 42

VARS = list(string.ascii_uppercase)
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '|'] 

VOCAB = {
    'pad': 0,
    'end': 1,
    **{v: i + 2 for i, v in enumerate(VARS)},
    **{s: i + 2 + len(VARS) for i, s in enumerate(SPECIALS)}
}

# --- Core Logic Simulation (Approach 1) ---

def simulate_derivation(start_facts: Set[str], rules: List[Tuple[str, str]]) -> List[Tuple[int, str]]:
    """
    Simulates the logic flow to determine the explicit depth of every derived fact.
    This ensures the target is always sorted: Step 1 -> Step 2 -> Step 3.
    """
    known = {f: 0 for f in start_facts}  # var -> depth
    derived = []
    
    changed = True
    max_iterations = 20  # Safety limit against cycles
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for source, target in rules:
            # Skip if target already known
            if target in known:
                continue
            
            # Can we fire this rule?
            if source in known:
                depth = known[source] + 1
                known[target] = depth
                derived.append((depth, target))
                changed = True
    
    # Sort by depth (primary) and name (secondary) for deterministic targets
    derived.sort(key=lambda x: (x[0], x[1]))
    return derived

def generate_improved_chain(config: DataProcessConfig) -> Tuple[str, str]:
    """
    Generates a sample with 'Hard Negatives' and meaningful structure.
    """
    chain_len = random.randint(config.min_steps, config.max_steps)
    available_vars = VARS[:config.num_vars]
    
    # 1. Setup Starting Facts
    if config.multi_start and chain_len >= 3:
        num_start = random.randint(1, 2)
    else:
        num_start = 1
    
    required_vars = chain_len + num_start
    if len(available_vars) < required_vars:
        available_vars = VARS
    
    # 2. Select Variables & Build True Chain
    chain_vars = random.sample(available_vars, required_vars)
    start_facts = set(chain_vars[:num_start])
    derivable = chain_vars[num_start:]
    
    true_rules = []
    
    # Link start facts to the first derived fact
    if num_start > 0:
        first_source = random.choice(list(start_facts))
        true_rules.append((first_source, derivable[0]))
    
    # Link the rest of the chain (A->B, B->C)
    for i in range(len(derivable) - 1):
        true_rules.append((derivable[i], derivable[i + 1]))
    
    # 3. Create Intelligent Distractors
    used_vars = set(chain_vars)
    unused_vars = [v for v in available_vars if v not in used_vars]
    distractor_rules = []
    
    # Type A: Dead Ends (Source exists, Target leads nowhere)
    # These force the model to check if a path continues.
    num_dead_end = config.num_distractors // 2
    for _ in range(num_dead_end):
        if not unused_vars: break
        source = random.choice(list(start_facts | set(derivable)))
        target = random.choice(unused_vars)
        distractor_rules.append((source, target))
        unused_vars.remove(target)
    
    # Type B: Blocked Rules (Source never becomes true)
    # These force the model to verify prerequisites.
    num_blocked = config.num_distractors - len(distractor_rules)
    for _ in range(num_blocked):
        if len(unused_vars) < 2: break
        source, target = random.sample(unused_vars, 2)
        distractor_rules.append((source, target))
    
    # 4. Generate Ground Truth via Simulation
    all_rules = true_rules + distractor_rules
    derived_with_depth = simulate_derivation(start_facts, all_rules)
    ground_truth = [var for depth, var in derived_with_depth]
    
    # 5. Format Output
    random.shuffle(all_rules)
    rule_strs = [f"{s}>{t}" for s, t in all_rules]
    start_facts_str = " ".join(sorted(list(start_facts)))
    
    input_str = f"Facts: {start_facts_str} | Rules: {' '.join(rule_strs)} | Target:"
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
                    if p in VOCAB: 
                        tokens.append(VOCAB[p])
    
    if len(tokens) > seq_len - 1: 
        return None
    
    tokens.append(VOCAB['end'])
    tokens += [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens

# --- Dataset Generation with Curriculum Sorting ---

def convert_subset(set_name, config, num):
    results = {k: [] for k in ["inputs", "labels"]}
    
    valid = 0
    pbar = tqdm(total=num, desc=f"Generating {set_name}")
    
    while valid < num:
        inp, targ = generate_improved_chain(config)
        
        # Combine input and target for autoregressive training
        full = f"{inp} {targ}"
        
        it = tokenize(inp, config.seq_len)
        ft = tokenize(full, config.seq_len)
        
        if it and ft:
            results["inputs"].append(np.array(it))
            results["labels"].append(np.array(ft))
            valid += 1
            pbar.update(1)
            
    pbar.close()

    # --- OPTIMIZATION: Curriculum Sorting ---
    # Sort samples by the length of the solution (target complexity).
    # This helps the model converge faster by learning simple implications first.
    print(f"Sorting {set_name} by complexity (Curriculum Learning)...")
    
    # Calculate solution length (approximation based on tokens)
    lengths = [np.count_nonzero(lbl) for lbl in results["labels"]]
    sort_indices = np.argsort(lengths)
    
    final_inputs = np.stack(results["inputs"])[sort_indices]
    final_labels = np.stack(results["labels"])[sort_indices]
    
    num_samples = len(final_inputs)
    final_puzzle_identifiers = np.arange(num_samples, dtype=np.int32)
    final_puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)
    final_group_indices = np.arange(num_samples + 1, dtype=np.int32)

    # Save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    np.save(os.path.join(save_dir, "all__inputs.npy"), final_inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), final_labels)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), final_puzzle_identifiers)
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), final_puzzle_indices)
    np.save(os.path.join(save_dir, "all__group_indices.npy"), final_group_indices)
    
    # Metadata
    metadata = {
        "seq_len": config.seq_len,
        "vocab_size": len(VOCAB),
        "pad_id": VOCAB['pad'],
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "num_puzzle_identifiers": num_samples,
        "total_groups": num_samples,
        "mean_puzzle_examples": 1.0,
        "total_puzzles": num_samples,
        "sets": ["all"]
    }

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    if set_name == "train":
        with open(os.path.join(config.output_dir, "vocab.json"), "w") as f: 
            json.dump(VOCAB, f)
        
        print("\n" + "="*60)
        print("EXAMPLE SAMPLE (Curriculum-Optimized):")
        print("="*60)
        print(f"INPUT:  {inp}")
        print(f"TARGET: {targ}")
        print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Generate Logic Chain Dataset")
    parser.add_argument("--output-dir", type=str, default="data/logic_chain")
    parser.add_argument("--num-train", type=int, default=20000)
    parser.add_argument("--num-test", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = DataProcessConfig(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Generating Optimized Logic Chain Data")
    print(f"Approach: Explicit Depth Simulation + Semantic Distractors + Curriculum Sorting")
    
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)
    
    print(f"Done. Data saved to {config.output_dir}")

if __name__ == "__main__":
    main()