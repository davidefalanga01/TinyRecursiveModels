import os
import json
import string
import random
import itertools
import numpy as np
from typing import Optional
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm


# Vocabulary for Propositional Logic
VARS = list(string.ascii_lowercase)
OPS = ['&', '|', '>']

VOCAB = {
    'pad': 0,
    **{v: i+1 for i, v in enumerate(VARS)},
    '&': 27, '|': 28, '>': 29, '~': 30,
    '(': 31, ')': 32, '|-': 33,
    'end': 34
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic"
    seq_len: int = 64
    num_train: int = 10000
    num_test: int = 1000
    num_vars: int = 4
    max_depth: int = 2
    subsample_size: Optional[int] = None
    seed: int = 42


def solve(premises_str, conclusion_str):
    """
    Checks if premises entail conclusion using truth tables.
    """
    # Parse unique variables
    full_expr = premises_str + " " + conclusion_str
    used_vars = sorted(list(set([c for c in full_expr if c in VARS])))
    
    if not used_vars:
        return False # Should not happen

    # Generate truth table
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        
        # Evaluate premises
        # Replace operators with python equivalents for eval
        # & -> and, | -> or, > -> <= (implication), ~ -> not
        # We need to be careful with eval security, but inputs are controlled here.
        
        def eval_expr(expr):
            py_expr = expr.replace('&', ' and ').replace('|', ' or ').replace('>', ' <= ').replace('~', ' not ')
            return eval(py_expr, {}, env)

        try:
            prem_val = eval_expr(premises_str)
            if prem_val:
                conc_val = eval_expr(conclusion_str)
                if not conc_val:
                    return False # Counter-example found: Premises True, Conclusion False
        except:
             return False # Malformed

    return True 

def tokenize(text, seq_len=64):
    # Tokenize
    tokens = [VOCAB.get(c, VOCAB.get(text[i:i+2], 0)) for i, c in enumerate(text)]
    # Fix double char token for |-
    clean_tokens = []
    skip = False
    for i, c in enumerate(text):
        if skip:
            skip = False
            continue
        if text[i:i+2] == '|-':
            clean_tokens.append(VOCAB['|-'])
            skip = True
        elif c in VOCAB:
            clean_tokens.append(VOCAB[c])
    
    if len(clean_tokens) > seq_len - 1:
        return None
        
    # Pad
    clean_tokens.append(VOCAB['end'])
    tokens = clean_tokens + [VOCAB['pad']] * (seq_len - len(clean_tokens))
    return tokens

def generate_random_expr(vocab_subset, depth=0, max_depth=2):
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return random.choice(vocab_subset)
    
    op = random.choice(OPS + ['~'])
    if op == '~':
        return f"~({generate_random_expr(vocab_subset, depth+1, max_depth)})"
    else:
        left = generate_random_expr(vocab_subset, depth+1, max_depth)
        right = generate_random_expr(vocab_subset, depth+1, max_depth)
        return f"({left}{op}{right})"

def generate_sample(seq_len=64, num_vars=4, max_depth=2):
    vocab_subset = VARS[:num_vars]
    while True:
        # Generate premises (randomly 1 to 3)
        num_premises = random.randint(1, 3)
        premises = [generate_random_expr(vocab_subset, 0, max_depth) for _ in range(num_premises)]
        premises_str = "(" + ")&(".join(premises) + ")"
        
        # Generate conclusion
        # valid/invalid balance
        if random.random() < 0.5:
            # Try to generate a valid one (likely related to premises)
            # Simple heuristic: pick a sub-expression or variation
            conclusion_str = random.choice(premises) if random.random() < 0.3 else generate_random_expr(vocab_subset, 0, max_depth)
        else:
            conclusion_str = generate_random_expr(vocab_subset, 0, max_depth)

        # Generate only valid label-text pairs
        if solve(premises_str, conclusion_str):
            label = conclusion_str 
        else:
            continue
        text = premises_str

        premise_tokenized = tokenize(text)
        conclusion_tokenized = tokenize(label)
        if (premise_tokenized is None) or (conclusion_tokenized is None):
            continue # Try again if too long
        
        return np.array(premise_tokenized), np.array(conclusion_tokenized)

# Dataset generation
def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    np.random.seed(config.seed)
    random.seed(config.seed)

    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for _ in tqdm(range(num_samples), desc=f"Generating {set_name}"):
        inp, out = generate_sample(config.seq_len, config.num_vars, config.max_depth)
       
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)

    # Convert to numpy arrays
    results = {
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
        "total_groups": len(results["group_indices"]) - 1,
        "mean_puzzle_examples": example_id / puzzle_id,
        "total_puzzles": puzzle_id,
        "sets": ["all"]
    }

    # Save dataset
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()
