"""
Propositional Logic SAT Task for TRM
Iteratively assigns truth values to forced variables
"""

import os, json, string, random, itertools, numpy as np
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# Minimal vocabulary (21 tokens vs your 35)
VARS = list(string.ascii_lowercase)[:8]
VOCAB = {
    'pad': 0,
    **{v: i+1 for i, v in enumerate(VARS)},  # 1-8: a-h
    '&': 9, '|': 10, '>': 11, '~': 12,
    '(': 13, ')': 14, '#': 15,
    'T': 16, 'F': 17, '=': 18, ',': 19,
    'end': 20
}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic"
    seq_len: int = 128
    num_train: int = 50000
    num_test: int = 5000
    num_vars: int = 4
    max_depth: int = 2
    seed: int = 42

def eval_logic_expr(expr_str: str, env: Dict[str, bool]) -> Optional[bool]:
    """Evaluate logical expression with variable assignments."""
    try:
        py_expr = (expr_str.replace('&', ' and ').replace('|', ' or ')
                           .replace('>', ' <= ').replace('~', ' not '))
        return eval(py_expr, {"__builtins__": {}}, env)
    except:
        return None

def get_vars_in_expr(expr: str) -> List[str]:
    return sorted(list(set([c for c in expr if c in VARS])))

def is_satisfiable(expr_str: str) -> bool:
    used_vars = get_vars_in_expr(expr_str)
    if not used_vars:
        return False
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        if eval_logic_expr(expr_str, env) is True:
            return True
    return False

def get_forced_variables(premises_str: str) -> Dict[str, bool]:
    """Determine which variables MUST have specific values."""
    used_vars = get_vars_in_expr(premises_str)
    if not used_vars:
        return {}
    
    forced = {}
    satisfying = []
    
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        if eval_logic_expr(premises_str, env) is True:
            satisfying.append(env)
    
    if not satisfying:
        return {}
    
    # Variable is forced if same value in ALL satisfying assignments
    for var in used_vars:
        values = [a[var] for a in satisfying]
        if all(values):
            forced[var] = True
        elif not any(values):
            forced[var] = False
    
    return forced

def generate_random_expr(vocab_subset: List[str], depth: int = 0, max_depth: int = 2) -> str:
    if depth >= max_depth or (depth > 0 and random.random() < 0.4):
        return random.choice(vocab_subset)
    
    op = random.choice(['&', '|', '>', '~'])
    if op == '~':
        return f"~({generate_random_expr(vocab_subset, depth+1, max_depth)})"
    else:
        left = generate_random_expr(vocab_subset, depth+1, max_depth)
        right = generate_random_expr(vocab_subset, depth+1, max_depth)
        return f"({left}{op}{right})"

def generate_premise_with_forced_vars(vocab_subset: List[str], max_depth: int) -> Tuple[str, Dict[str, bool]]:
    """Generate premises that force at least 2 variables."""
    for _ in range(100):
        # Create facts and implication chains
        num_facts = random.randint(1, 2)
        fact_vars = random.sample(vocab_subset, min(num_facts, len(vocab_subset)))
        
        clauses = []
        for var in fact_vars:
            clauses.append(var if random.random() < 0.7 else f"~({var})")
        
        # Add implications
        remaining = [v for v in vocab_subset if v not in fact_vars]
        if remaining and fact_vars:
            for _ in range(random.randint(1, 3)):
                ant = random.choice(fact_vars)
                cons = random.choice(remaining)
                clauses.append(f"({ant}>{cons})")
        
        premises_str = "&".join(f"({c})" for c in clauses)
        
        if not is_satisfiable(premises_str):
            continue
        
        forced = get_forced_variables(premises_str)
        
        # Quality: Need ≥2 forced vars, at least 1 derived (not just facts)
        if len(forced) >= 2:
            derived = [v for v in forced if v not in fact_vars]
            if derived:
                return premises_str, forced
    
    # Fallback
    v1, v2 = vocab_subset[0], vocab_subset[1]
    premises = f"({v1})&({v1}>{v2})"
    return premises, get_forced_variables(premises)

def tokenize(text: str, seq_len: int) -> Optional[np.ndarray]:
    tokens = []
    i = 0
    while i < len(text):
        if text[i] in VOCAB:
            tokens.append(VOCAB[text[i]])
            i += 1
        else:
            i += 1
    
    if len(tokens) > seq_len - 1:
        return None
    
    tokens.append(VOCAB['end'])
    tokens.extend([VOCAB['pad']] * (seq_len - len(tokens)))
    return np.array(tokens, dtype=np.int32)

def generate_sample(seq_len: int, num_vars: int, max_depth: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    vocab_subset = VARS[:num_vars]
    premises, forced = generate_premise_with_forced_vars(vocab_subset, max_depth)
    
    if not forced:
        return None, None
    
    # Create assignment sequence
    sorted_vars = sorted(forced.keys())
    assignments = [f"{v}={'T' if forced[v] else 'F'}" for v in sorted_vars]
    
    input_str = f"{premises}#"
    output_str = f"{premises}#{','.join(assignments)}"
    
    inp = tokenize(input_str, seq_len)
    out = tokenize(output_str, seq_len)
    
    return (inp, out) if inp is not None and out is not None else (None, None)

def generate_dataset(set_name: str, config: DataProcessConfig, num_samples: int):
    np.random.seed(config.seed if set_name == "train" else config.seed + 1)
    random.seed(config.seed if set_name == "train" else config.seed + 1)
    
    results = {"inputs": [], "labels": [], "puzzle_indices": [0], 
               "group_indices": [0], "puzzle_identifiers": []}
    
    pbar = tqdm(total=num_samples, desc=f"Generating {set_name}")
    example_id = puzzle_id = 0
    
    while example_id < num_samples:
        inp, out = generate_sample(config.seq_len, config.num_vars, config.max_depth)
        if inp is None:
            continue
        
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)
        pbar.update(1)
    
    pbar.close()
    
    dataset = {k: np.stack(v) if k in ["inputs", "labels"] else np.array(v, dtype=np.int32) 
               for k, v in results.items()}
    
    metadata = {
        "seq_len": config.seq_len, "vocab_size": len(VOCAB), "pad_id": 0,
        "ignore_label_id": 0, "blank_identifier_id": 0, "num_puzzle_identifiers": 1,
        "total_groups": puzzle_id, "mean_puzzle_examples": 1.0,
        "total_puzzles": puzzle_id, "sets": ["all"]
    }
    
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    for k, v in dataset.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{set_name}: {num_samples} samples, vocab_size={len(VOCAB)}")

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([""], f)
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(VOCAB, f, indent=2)
    
    generate_dataset("train", config, config.num_train)
    generate_dataset("test", config, config.num_test)
    print(f"\n✓ Complete! Output: {config.output_dir}")

if __name__ == "__main__":
    cli()