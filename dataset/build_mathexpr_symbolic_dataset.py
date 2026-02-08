import os
import json
import random
import numpy as np
import re
import string
from typing import List, Dict
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

from common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100 # Standard pytorch ignore index

# Reserved tokens
OPS = ['+', '-', '*', '/']
LETTERS = list(string.ascii_uppercase) # A, B, C ... Z
SYMBOLS = ['pad', 'end', '(', ')'] + OPS + LETTERS

VOCAB_MAP = {s: i for i, s in enumerate(SYMBOLS)}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/symbolic_letters"
    seq_len: int = 64
    num_train: int = 20000
    num_test: int = 2000
    seed: int = 42
    max_depth: int = 3
    num_vars: int = 5 # Different letter to use 5 = A, B, C, D, E)
    max_answer_len: int = 20 # Max length of result

def tokenize_simple(expr: str):
    """Simple tokenizetion """
    expr = expr.replace("(", " ( ").replace(")", " ) ")
    return [t for t in expr.split() if t.strip()]

def parse_prefix(tokens):
    """
    Parse prefix notation and apply the symbolic rules on strings.
    """
    if not tokens:
        raise ValueError("Empty token list")
    
    tok = tokens.pop(0)
    

    if tok in LETTERS:
        return tok
    
    if tok == "(":
        op = tokens.pop(0)
        left = parse_prefix(tokens)
        right = parse_prefix(tokens)
        closing = tokens.pop(0)
        if closing != ")":
            raise ValueError(f"Expected ')', found {closing}")
        
        # (A + B) = AB (Concat)
        if op == "+": return left + right
        
        # (A - B) = BA (Inverse concat)
        if op == "-": return right + left
        
        # (A * B) = AAB (Double first + second)
        if op == "*": return left + left + right
        
        # (A / B) = BBA (Double second + first)
        if op == "/": return right + right + left
        
        raise ValueError(f"Invalid operator: {op}")
    
    raise ValueError(f"Unexpected token: {tok}")

def eval_prefix(expr: str) -> str:
    """Eval in a symbolic way the string and return the result."""
    tokens = tokenize_simple(expr)
    result = parse_prefix(tokens)
    return result

def generate_prefix(depth: int, distinct_vars: List[str]) -> str:
    # Case 0: random letter 
    if depth == 0 or random.random() < 0.2:
        return random.choice(distinct_vars)
    
    op = random.choice(OPS)
    
    left = generate_prefix(depth - 1, distinct_vars)
    right = generate_prefix(depth - 1, distinct_vars)
    
    return f"({op} {left} {right})"


def tokenize(expr: str, seq_len: int):
    # Find letters/operators
    tokens = re.findall(r'[A-Z]|[+\-*/()]', expr)
    
    token_ids = []
    
    for t in tokens:
        if t in VOCAB_MAP:
            token_ids.append(VOCAB_MAP[t])
        else:
            # Fallback if something goes wrong
            token_ids.append(VOCAB_MAP['pad'])

    # Append End Token
    token_ids.append(VOCAB_MAP['end'])
    
    # Padding
    pad_id = VOCAB_MAP['pad']
    if len(token_ids) < seq_len:
        pad_len = seq_len - len(token_ids)
        token_ids.extend([pad_id] * pad_len)
    else:
        # Truncate if longer
        token_ids = token_ids[:seq_len]
        token_ids[-1] = VOCAB_MAP['end']
        
    return np.array(token_ids, dtype=np.int32)


def convert_subset(name: str, config: DataProcessConfig):
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    example_id = 0
    puzzle_id = 0
    
    # Letter to use (es. ['A', 'B', 'C', 'D', 'E'])
    active_vars = LETTERS[:config.num_vars]

    total = config.num_train if name == "train" else config.num_test
    
    pbar = tqdm(total=total, desc=f"Generating {name}")
    
    while example_id < total:
        
        depth = random.randint(1, config.max_depth)
        expr = generate_prefix(depth, active_vars)
        
        # Compute the result
        try:
            result_str = eval_prefix(expr)
        except Exception:
            continue

        # Length check
        if len(result_str) > config.max_answer_len:
            continue
            
        inp = tokenize(expr, config.seq_len)
        
        label = np.full((config.seq_len,), IGNORE_LABEL_ID, dtype=np.int32)
        
        # Convert str to ids
        try:
            res_ids = [VOCAB_MAP[char] for char in result_str]
        except KeyError:
            continue
            
        # Example: Label = [ID_A, ID_B, ID_B, ID_A, END, -100, -100...]
        len_res = len(res_ids)
        if len_res >= config.seq_len:
            continue 

        label[:len_res] = res_ids
        if len_res < config.seq_len:
            label[len_res] = VOCAB_MAP['end']
        
        results["inputs"].append(inp)
        results["labels"].append(label)
        
        results["puzzle_identifiers"].append(0)
        example_id += 1
        puzzle_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(puzzle_id)
        
        pbar.update(1)

        # Debug print 
        if example_id == 1:
            print(f"\n--- Example {name} ---")
            print(f"Expression: {expr}")
            print(f"Result Str: {result_str}")
            print(f"Input Ids : {inp[:8]} ...")
            print(f"Label Ids : {label[:8]} ...")
            
            # Decode for check
            rev_map = {v:k for k,v in VOCAB_MAP.items()}
            
            decoded_in = [rev_map.get(i, '?') for i in inp if i != VOCAB_MAP['pad']]
            decoded_out = [rev_map.get(i, '?') for i in label if i != IGNORE_LABEL_ID]
            
            print(f"Decoded IN : {' '.join(decoded_in)}")
            print(f"Decoded OUT: {' '.join(decoded_out)}")
    
    pbar.close()
    
    # Convert to numpy
    results_np = {
        k: np.stack(v).astype(np.int32) if isinstance(v[0], np.ndarray) else np.array(v, dtype=np.int32)
        for k, v in results.items()
        if k not in ("puzzle_indices", "group_indices")
    }
    
    results_np["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
    results_np["group_indices"] = np.array(results["group_indices"], dtype=np.int32)

    # Save Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len,
        vocab_size=len(VOCAB_MAP), 
        pad_id=VOCAB_MAP['pad'],
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results_np["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results_np["puzzle_indices"]) - 1,
        sets=["all"]
    )

    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    for k, v in results_np.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Vocab size: {len(VOCAB_MAP)}")
    print(f"Vars used: {LETTERS[:config.num_vars]}")
    
    convert_subset("train", config)
    convert_subset("test", config)
    
    print("\nDataset generation complete.")

if __name__ == "__main__":
    cli()
