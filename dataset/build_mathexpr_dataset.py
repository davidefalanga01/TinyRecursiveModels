import os
import json
import random
import re
import numpy as np
from typing import List
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

from common import PuzzleDatasetMetadata

IGNORE_LABEL_ID = 0


NUM_TOKEN = "NUM"
OPS = ['+', '-', '*', '/']

VOCAB = {
    'pad': 0,
    'end': 1,
    NUM_TOKEN: 2,
    **{op: i + 3 for i, op in enumerate(OPS)},
    '(': 3 + len(OPS),
    ')': 4 + len(OPS),
}

TOKEN_TYPE = {
    'PAD': 0,
    'NUM': 1,
    'OP': 2,
    'PAREN': 3,
    'END': 4,
}

cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/arith_dataset"
    seq_len: int = 64
    num_train: int = 20000
    num_test: int = 2000
    seed: int = 42
    max_depth: int = 3
    min_number: int = 1
    max_number: int = 20



def generate_expression(depth: int, min_num: int, max_num: int) -> str:
    if depth == 0 or random.random() < 0.2:
        return str(random.randint(min_num, max_num))

    op = random.choice(OPS)
    left = generate_expression(depth - 1, min_num, max_num)
    right = generate_expression(depth - 1, min_num, max_num)

    # integer divisions only
    if op == '/':
        denom = random.randint(2, 9)
        num = denom * random.randint(1, 10)
        left, right = str(num), str(denom)

    return f"({left} {op} {right})"


def eval_expression(expr: str) -> int:
    return int(eval(expr))

def tokenize(expr: str, seq_len: int):
    token_ids: List[int] = []
    num_values: List[int] = []
    token_types: List[int] = []

    for t in re.findall(r'\d+|[+\-*/()]', expr):
        if t.isdigit():
            token_ids.append(VOCAB[NUM_TOKEN])
            num_values.append(int(t))
            token_types.append(TOKEN_TYPE['NUM'])
        elif t in OPS:
            token_ids.append(VOCAB[t])
            num_values.append(0)
            token_types.append(TOKEN_TYPE['OP'])
        else:
            token_ids.append(VOCAB[t])
            num_values.append(0)
            token_types.append(TOKEN_TYPE['PAREN'])

    # END token
    token_ids.append(VOCAB['end'])
    num_values.append(0)
    token_types.append(TOKEN_TYPE['END'])

    # pad/truncate
    pad_id = VOCAB['pad']
    if len(token_ids) < seq_len:
        pad_len = seq_len - len(token_ids)
        token_ids.extend([pad_id]*pad_len)
        num_values.extend([0]*pad_len)
        token_types.extend([TOKEN_TYPE['PAD']]*pad_len)
    else:
        token_ids = token_ids[:seq_len]
        token_ids[-1] = VOCAB['end']
        num_values = num_values[:seq_len]
        num_values[-1] = 0
        token_types = token_types[:seq_len]
        token_types[-1] = TOKEN_TYPE['END']

    # final input = concatenation (simple 3×seq_len)
    input_vec = np.stack([token_ids, num_values, token_types], axis=1).astype(np.int32)  
    return input_vec


def convert_subset(name: str, config: DataProcessConfig):
    rng = np.random.default_rng(config.seed)

    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
    }

    example_id = 0
    puzzle_id = 0

    num_samples = config.num_train if name == "train" else config.num_test

    for _ in tqdm(range(num_samples), desc=f"{name}"):
        depth = random.randint(1, config.max_depth)
        expr = generate_expression(depth, config.min_number, config.max_number)
        result = eval_expression(expr)

        # (seq_len, 3)
        inp = tokenize(expr, config.seq_len)

        # labels: 1 target per example
        # we place it at last position, others are ignore
        label = np.full((config.seq_len,), IGNORE_LABEL_ID, dtype=np.int32)
        label[-1] = result

        results["inputs"].append(inp.reshape(-1))
        results["labels"].append(label)
        results["puzzle_identifiers"].append(0)

        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["group_indices"].append(puzzle_id)

    # convert to arrays
    def _stack(x):
        return np.stack(x, axis=0).astype(np.int32)

    results_np = {
        "inputs": _stack(results["inputs"]),
        "labels": _stack(results["labels"]),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=config.seq_len * 3,
        vocab_size=len(VOCAB),
        pad_id=VOCAB['pad'],
        ignore_label_id=IGNORE_LABEL_ID,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results_np["group_indices"])-1,
        mean_puzzle_examples=1,
        total_puzzles=len(results_np["puzzle_indices"])-1,
        sets=["all"]
    )

    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    # save metadata
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # save arrays
    for key, val in results_np.items():
        np.save(os.path.join(save_dir, f"all__{key}.npy"), val)

    print(f"Example {name}: {expr} = {result}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)

    convert_subset("train", config)
    convert_subset("test", config)

    print("Arithmetic PuzzleDataset generation complete.")


if __name__ == "__main__":
    cli()
