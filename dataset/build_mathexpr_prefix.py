import os
import json
import random
import numpy as np
import re
from typing import List, Dict
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

from common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = -100 # Standard pytorch ignore index

# Reserved tokens
OPS = ['+', '-', '*', '/']
SYMBOLS = ['pad', 'end', '(', ')'] + OPS

# Static symbols to IDs 0-7
VOCAB_MAP = {s: i for i, s in enumerate(SYMBOLS)}

# e.g. Number 0 -> ID 1000, Number 22 -> ID 1022, Number -5 -> ID 995
NUM_OFFSET = 1000 

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/arith_prefix"
    seq_len: int = 64
    num_train: int = 20000
    num_test: int = 2000
    seed: int = 41
    max_depth: int = 3
    min_number: int = 1
    max_number: int = 20


def tokenize_simple(expr: str):
    expr = expr.replace("(", " ( ").replace(")", " ) ")
    return [t for t in expr.split() if t.strip()]

def parse_prefix(tokens):
    if not tokens:
        raise ValueError("Empty token list")
    
    tok = tokens.pop(0)
    
    if tok.isdigit() or (tok.startswith("-") and tok[1:].isdigit()):
        return int(tok)
    
    if tok == "(":
        op = tokens.pop(0)
        left = parse_prefix(tokens)
        right = parse_prefix(tokens)
        closing = tokens.pop(0)
        if closing != ")":
            raise ValueError(f"Expected ')', found {closing}")
        
        if op == "+": return left + right
        if op == "-": return left - right
        if op == "*": return left * right
        if op == "/": return int(left / right) # Integer division for simplicity
        
        raise ValueError(f"Invalid operator: {op}")
    
    raise ValueError(f"Unexpected token: {tok}")

def eval_prefix(expr: str) -> int:
    tokens = tokenize_simple(expr)
    result = parse_prefix(tokens)
    return int(result)

def generate_prefix(depth: int, min_num: int, max_num: int) -> str:
    if depth == 0 or random.random() < 0.3:
        return str(random.randint(min_num, max_num))
    
    op = random.choice(OPS)
    
    if op == '/':
        denom = random.randint(2, 9)
        num = denom * random.randint(1, 10)
        left = str(num)
        right = str(denom)
    else:
        left = generate_prefix(depth - 1, min_num, max_num)
        right = generate_prefix(depth - 1, min_num, max_num)
    
    return f"({op} {left} {right})"


def tokenize(expr: str, seq_len: int):
    # Splits into: '(', '+', '22', '3', ')'
    tokens = re.findall(r'\d+|[+\-*/()]', expr)
    
    token_ids = []
    
    for t in tokens:
        if t in VOCAB_MAP:
            # It is an operator or parenthesis
            token_ids.append(VOCAB_MAP[t])
        elif t.isdigit():
            # It is a number, map it to ID space
            token_ids.append(int(t) + NUM_OFFSET)
        else:
            # Fallback (should not happen with this regex)
            token_ids.append(VOCAB_MAP['pad'])

    # Append End Token
    token_ids.append(VOCAB_MAP['end'])
    
    # Padding
    pad_id = VOCAB_MAP['pad']
    if len(token_ids) < seq_len:
        pad_len = seq_len - len(token_ids)
        token_ids.extend([pad_id] * pad_len)
    else:
        # Truncate if too long
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
    
    # Track max token ID seen to define vocab size in metadata
    max_id_seen = NUM_OFFSET

    total = config.num_train if name == "train" else config.num_test
    
    for _ in tqdm(range(total), desc=f"Generating {name}"):
        
        depth = random.randint(1, config.max_depth)
        expr = generate_prefix(depth, config.min_number, config.max_number)
        
        # Calculate result (Label)
        try:
            result = eval_prefix(expr)
        except Exception:
            # Skip invalid generations (div by zero etc)
            continue

        # Tokenize Input
        inp = tokenize(expr, config.seq_len)
        
        # We assume the task is to predict the result at the last position
        label = np.full((config.seq_len,), IGNORE_LABEL_ID, dtype=np.int32)
        
        # Map the numeric result to a Token ID
        result_token_id = result + NUM_OFFSET
        label[0] = result_token_id
        label[1] = 1
        
        # Update max ID for metadata
        current_max = max(np.max(inp), result_token_id)
        if current_max > max_id_seen:
            max_id_seen = current_max

        results["inputs"].append(inp)
        results["labels"].append(label)
        
        results["puzzle_identifiers"].append(0)
        example_id += 1
        puzzle_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(puzzle_id)

        # Print first example for verification
        if example_id == 1:
            print(f"\n--- Example {name} ---")
            print(f"Expression: {expr} = {result}")
            print(f"Input Ids : {inp[:5]} ...")
            print(f"Label Ids : {label[:5]}")
            # Decode back for sanity check
            decoded = []
            rev_map = {v:k for k,v in VOCAB_MAP.items()}
            for i in inp:
                if i in rev_map: decoded.append(rev_map[i])
                elif i == 0: decoded.append('PAD')
                else: decoded.append(str(i - NUM_OFFSET))
            print(f"Decoded   : {' '.join(decoded[:10])} ...")
    
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
        vocab_size=int(max_id_seen + 100), 
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
    
    convert_subset("train", config)
    convert_subset("test", config)
    
    print("\nDataset generation complete.")

if __name__ == "__main__":
    cli()
    print(tokenize("( + 21 5 )", 10))
