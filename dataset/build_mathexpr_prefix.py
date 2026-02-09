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

IGNORE_LABEL_ID = -100 

# Reserved tokens
OPS = ['+', '-', '*', '/']
SYMBOLS = ['pad', 'end', '(', ')'] + OPS
VOCAB_MAP = {s: i for i, s in enumerate(SYMBOLS)}
NUM_OFFSET = 100

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/arith_prefix_small"
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
        if op == "/": return int(left / right)
        
        raise ValueError(f"Invalid operator: {op}")
    
    raise ValueError(f"Unexpected token: {tok}")

def eval_prefix(expr: str) -> int:
    tokens = tokenize_simple(expr)
    result = parse_prefix(tokens)
    return int(result)

# Top-Down logic for generation
def get_factors(n):
    """Find factors for multiplication."""
    factors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            factors.append(i)
            if i*i != n:
                factors.append(n // i)
    return factors

def generate_constrained_prefix(target: int, depth: int, min_val: int, max_val: int) -> str:
    """
    Generate an expression that is = to 'target';
    this ensure that there is none value bigger than 'max_val'.
    """
    if depth == 0 or random.random() < 0.3:
        return str(target)

    ops = list(OPS)
    random.shuffle(ops)

    for op in ops:
        left_val, right_val = None, None

        if op == '+':
            if target - min_val < min_val: continue 
            left_val = random.randint(min_val, target - min_val)
            right_val = target - left_val

        elif op == '-':
            limit_R = max_val - target
            if limit_R < min_val: continue 
            right_val = random.randint(min_val, limit_R)
            left_val = target + right_val

        elif op == '*':
            # target = L * R
            factors = get_factors(target)
            valid_factors = [f for f in factors if f >= min_val and (target // f) >= min_val]
            if not valid_factors: continue
            
            # Attempting to find values different from 1 to avoid trivial solution
            non_one = [f for f in valid_factors if f != 1 and (target // f) != 1]
            if non_one:
                left_val = random.choice(non_one)
            else:
                if random.random() < 0.5: continue 
                left_val = random.choice(valid_factors)
            
            right_val = target // left_val

        elif op == '/':
            limit_R = max_val // target
            if limit_R < min_val: continue
            
            if limit_R == 1 and min_val == 1:
                 if random.random() < 0.8: continue

            right_val = random.randint(min_val, limit_R)
            left_val = target * right_val

        if left_val is not None and right_val is not None:
            s_left = generate_constrained_prefix(left_val, depth - 1, min_val, max_val)
            s_right = generate_constrained_prefix(right_val, depth - 1, min_val, max_val)
            return f"({op} {s_left} {s_right})"

    # Fallback if none of operators satisfies the constraints
    return str(target)


def tokenize(expr: str, seq_len: int):
    tokens = re.findall(r'\d+|[+\-*/()]', expr)
    token_ids = []
    
    for t in tokens:
        if t in VOCAB_MAP:
            token_ids.append(VOCAB_MAP[t])
        elif t.isdigit():
            token_ids.append(int(t) + NUM_OFFSET)
        else:
            token_ids.append(VOCAB_MAP['pad'])

    token_ids.append(VOCAB_MAP['end'])
    
    pad_id = VOCAB_MAP['pad']
    if len(token_ids) < seq_len:
        pad_len = seq_len - len(token_ids)
        token_ids.extend([pad_id] * pad_len)
    else:
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
    max_id_seen = NUM_OFFSET

    total = config.num_train if name == "train" else config.num_test
    
    for _ in tqdm(range(total), desc=f"Generating {name}"):
        target_result = random.randint(config.min_number, config.max_number)
        
        # Generate tree expression to guarantee sanity
        depth = random.randint(1, config.max_depth)
        expr = generate_constrained_prefix(target_result, depth, config.min_number, config.max_number)
        
        try:
            check_val = eval_prefix(expr)
            if check_val != target_result:
                print(f"Error: Generated {expr} = {check_val}, expected {target_result}")
                continue
        except Exception as e:
            print(f"Error parsing {expr}: {e}")
            continue

        inp = tokenize(expr, config.seq_len)
        
        label = np.full((config.seq_len,), IGNORE_LABEL_ID, dtype=np.int32)
        result_token_id = target_result + NUM_OFFSET
        label[0] = result_token_id
        label[1] = 1 
        
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

        if example_id == 1:
            print(f"\n--- Example {name} ---")
            print(f"Expression: {expr} = {target_result}")
            print(f"Input Ids : {inp[:10]} ...")
            print(f"Label Ids : {label[:5]}")
            decoded = []
            rev_map = {v:k for k,v in VOCAB_MAP.items()}
            for i in inp:
                if i in rev_map: decoded.append(rev_map[i])
                elif i == 0: decoded.append('PAD')
                else: decoded.append(str(i - NUM_OFFSET))
            print(f"Decoded   : {' '.join(decoded[:15])} ...")
    
    results_np = {
        k: np.stack(v).astype(np.int32) if isinstance(v[0], np.ndarray) else np.array(v, dtype=np.int32)
        for k, v in results.items()
        if k not in ("puzzle_indices", "group_indices")
    }
    
    results_np["puzzle_indices"] = np.array(results["puzzle_indices"], dtype=np.int32)
    results_np["group_indices"] = np.array(results["group_indices"], dtype=np.int32)

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
    
    print(f"Generating expressions with values in range [{config.min_number}, {config.max_number}]")
    convert_subset("train", config)
    convert_subset("test", config)
    
    print("\nDataset generation complete.")

if __name__ == "__main__":
    cli()
