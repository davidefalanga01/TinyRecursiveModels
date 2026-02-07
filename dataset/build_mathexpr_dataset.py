import os
import json
import random
import re
import numpy as np
from typing import List
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm


NUM_TOKEN = "NUM"
OPS = ['+', '-', '*', '/']
PARENS = ['(', ')']

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

    # divisioni sempre intere
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

        else:  # parentesi
            token_ids.append(VOCAB[t])
            num_values.append(0)
            token_types.append(TOKEN_TYPE['PAREN'])

    # END token
    token_ids.append(VOCAB['end'])
    num_values.append(0)
    token_types.append(TOKEN_TYPE['END'])

    # padding / truncation
    if len(token_ids) < seq_len:
        pad_len = seq_len - len(token_ids)
        token_ids.extend([VOCAB['pad']] * pad_len)
        num_values.extend([0] * pad_len)
        token_types.extend([TOKEN_TYPE['PAD']] * pad_len)
    else:
        token_ids = token_ids[:seq_len]
        token_ids[-1] = VOCAB['end']
        num_values = num_values[:seq_len]
        num_values[-1] = 0
        token_types = token_types[:seq_len]
        token_types[-1] = TOKEN_TYPE['END']

    return (
        np.array(token_ids, dtype=np.int64),
        np.array(num_values, dtype=np.int64),
        np.array(token_types, dtype=np.int64),
    )

def generate_sample(config: DataProcessConfig):
    depth = random.randint(1, config.max_depth)
    expr = generate_expression(
        depth,
        config.min_number,
        config.max_number
    )
    result = eval_expression(expr)
    return expr, result


def convert_subset(name: str, config: DataProcessConfig, num_samples: int):
    tokens, numbers, types, labels = [], [], [], []

    for _ in tqdm(range(num_samples), desc=f"Generating {name}"):
        expr, result = generate_sample(config)
        tok, num, typ = tokenize(expr, config.seq_len)

        tokens.append(tok)
        numbers.append(num)
        types.append(typ)
        labels.append(np.array([result], dtype=np.int64))

    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "tokens.npy"), np.stack(tokens))
    np.save(os.path.join(save_dir, "numbers.npy"), np.stack(numbers))
    np.save(os.path.join(save_dir, "types.npy"), np.stack(types))
    np.save(os.path.join(save_dir, "labels.npy"), np.stack(labels))

    metadata = {
        "seq_len": config.seq_len,
        "vocab_size": len(VOCAB),
        "num_token_id": VOCAB[NUM_TOKEN],
        "pad_id": VOCAB['pad'],
        "num_samples": num_samples,
    }

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Example [{name}]: {expr} = {result}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    random.seed(config.seed)
    np.random.seed(config.seed)

    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

    print("✔ Dataset generation complete.")


if __name__ == "__main__":
    cli()
