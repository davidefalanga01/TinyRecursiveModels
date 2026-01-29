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
    output_dir: str = "data/logic_mixed"
    seq_len: int = 64
    num_train: int = 10000
    num_test: int = 1000
    num_vars: int = 4
    max_depth: int = 2
    subsample_size: Optional[int] = None
    seed: int = 42
    min_nontrivial_ratio: float = 0.5  # NEW: Minimum ratio of non-trivial samples
    max_attempts_per_sample: int = 100  # NEW: Max attempts before giving up


def eval_logic_expr(expr_str, env):
    """
    Safely evaluate a logical expression given variable assignments.
    Returns None if evaluation fails.
    """
    try:
        py_expr = expr_str.replace('&', ' and ').replace('|', ' or ').replace('>', ' <= ').replace('~', ' neg ')
        local_env = {'neg': lambda x: not x}
        return eval(py_expr, {}, {**env, **local_env})
    except:
        return None


def get_vars_in_expr(expr):
    """Extract all variables used in an expression."""
    return sorted(list(set([c for c in expr if c in VARS])))


def is_satisfiable(expr_str):
    """Check if an expression is satisfiable (has at least one model)."""
    used_vars = get_vars_in_expr(expr_str)
    if not used_vars:
        return False
    
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        result = eval_logic_expr(expr_str, env)
        if result is True:
            return True
    return False


def is_tautology(expr_str):
    """Check if an expression is a tautology (true in all models)."""
    used_vars = get_vars_in_expr(expr_str)
    if not used_vars:
        return False
    
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        result = eval_logic_expr(expr_str, env)
        if result is False or result is None:
            return False
    return True


def get_forced_variables(premises_str):
    """
    Determine which variables are forced to specific values by the premises.
    Returns dict: {var: bool} for forced variables, empty dict if none forced.
    """
    used_vars = get_vars_in_expr(premises_str)
    if not used_vars:
        return {}
    
    forced = {}
    for var in used_vars:
        values_when_true = []
        values_when_false = []
        
        # Check if var is always True or always False in satisfying models
        for values in itertools.product([False, True], repeat=len(used_vars)):
            env = dict(zip(used_vars, values))
            if eval_logic_expr(premises_str, env):
                if env[var]:
                    values_when_true.append(True)
                else:
                    values_when_false.append(True)
        
        # If var is never False in satisfying models, it's forced to True
        if values_when_true and not values_when_false:
            forced[var] = True
        # If var is never True in satisfying models, it's forced to False
        elif values_when_false and not values_when_true:
            forced[var] = False
    
    return forced


def entails(premises_str, conclusion_str):
    """
    Check if premises entail conclusion using truth tables.
    Returns True if every model satisfying premises also satisfies conclusion.
    """
    full_expr = premises_str + " " + conclusion_str
    used_vars = get_vars_in_expr(full_expr)
    
    if not used_vars:
        return False
    
    for values in itertools.product([False, True], repeat=len(used_vars)):
        env = dict(zip(used_vars, values))
        
        prem_val = eval_logic_expr(premises_str, env)
        if prem_val is True:
            conc_val = eval_logic_expr(conclusion_str, env)
            if conc_val is False or conc_val is None:
                return False  # Counter-example found
    
    return True


def tokenize(text, seq_len=64):
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
    
    clean_tokens.append(VOCAB['end'])
    tokens = clean_tokens + [VOCAB['pad']] * (seq_len - len(clean_tokens))
    return tokens


def generate_random_expr(vocab_subset, depth=0, max_depth=2):
    """Generate a random logical expression."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        return random.choice(vocab_subset)
    
    op = random.choice(OPS + ['~'])
    if op == '~':
        return f"~({generate_random_expr(vocab_subset, depth+1, max_depth)})"
    else:
        left = generate_random_expr(vocab_subset, depth+1, max_depth)
        right = generate_random_expr(vocab_subset, depth+1, max_depth)
        return f"({left}{op}{right})"


def generate_conclusion_from_inference(premises_str, vocab_subset, forced_vars):
    """
    Generate a non-trivial conclusion using inference patterns.
    This creates conclusions that require actual reasoning.
    """
    strategies = []
    
    # Strategy 1: Combine forced variables with implications
    if len(forced_vars) >= 2:
        vars_list = list(forced_vars.keys())
        v1, v2 = random.sample(vars_list, 2)
        # Create implication chain
        strategies.append(f"({v1}>{v2})")
        strategies.append(f"({v1}&{v2})")
    
    # Strategy 2: Use negation of forced variables in complex ways
    if forced_vars:
        var = random.choice(list(forced_vars.keys()))
        possible_others = [v for v in vocab_subset if v not in forced_vars]
        if possible_others:
            other_var = random.choice(possible_others)
            strategies.append(f"(~({var})>{other_var})")
            strategies.append(f"({var}|~({other_var}))")
    
    # Strategy 3: Implication with free variables
    free_vars = [v for v in vocab_subset if v not in forced_vars]
    if free_vars and forced_vars:
        free_var = random.choice(free_vars)
        forced_var = random.choice(list(forced_vars.keys()))
        strategies.append(f"({free_var}>{forced_var})")
        strategies.append(f"(~({free_var})|{forced_var})")
    
    # Strategy 4: Conjunction of forced with implication
    if len(forced_vars) >= 1:
        forced_var = random.choice(list(forced_vars.keys()))
        other_var = random.choice(vocab_subset)
        strategies.append(f"(({forced_var}>{other_var})|{forced_var})")
    
    # Strategy 5: Nested structure requiring reasoning
    if len(vocab_subset) >= 3:
        v1, v2, v3 = random.sample(vocab_subset, 3)
        strategies.append(f"(({v1}>{v2})>({v2}>{v3}))")
        strategies.append(f"(({v1}&{v2})|({v2}&{v3}))")
    
    if not strategies:
        # Fallback: simple random expression
        return generate_random_expr(vocab_subset, 0, 2)
    
    return random.choice(strategies)


def check_triviality(premises_str, conclusion_str, all_premises, forced_vars):
    """
    Enhanced triviality check. Returns (is_trivial, reason).
    """
    # Check 1: Tautology
    if is_tautology(conclusion_str):
        return True, "tautology"
    
    # Check 2: Identity (input == output)
    if premises_str == conclusion_str:
        return True, "identity"
    
    # Check 3: Single-premise entailment
    for p in all_premises:
        if entails(p, conclusion_str):
            return True, "single_premise"
    
    # Check 4: All variables forced (makes reasoning trivial)
    conclusion_vars = get_vars_in_expr(conclusion_str)
    if conclusion_vars and all(v in forced_vars for v in conclusion_vars):
        # All conclusion variables are forced - check if it's just extracting them
        # Allow if there's actual logical structure (implications, negations)
        if '>' not in conclusion_str and '~' not in conclusion_str:
            return True, "all_vars_forced"
    
    # Check 5: Direct variable extraction (conclusion is subset of premises)
    # This is a simple heuristic: if conclusion is just (var1 & var2 & ...)
    # and all these vars appear as facts in premises
    if conclusion_str.replace('(', '').replace(')', '').replace('&', '').replace(' ', '') in \
       premises_str.replace('(', '').replace(')', '').replace('&', '').replace(' ', ''):
        # Substring match - too simple
        if '|' not in conclusion_str and '>' not in conclusion_str and '~' not in conclusion_str:
            return True, "direct_extraction"
    
    return False, "non_trivial"


def generate_sample(seq_len=64, num_vars=4, max_depth=2, require_nontrivial=True):
    """
    Generate a single sample with improved non-triviality guarantees.
    """
    vocab_subset = VARS[:num_vars]
    
    max_attempts = 50
    for attempt in range(max_attempts):
        # 1. Generate premises with mix of rules and facts
        num_facts = random.randint(0, 2)  # Allow 0 facts for more variety
        facts = [random.choice(vocab_subset) for _ in range(num_facts)]
        
        num_rules = random.randint(1, 3)
        rules = [generate_random_expr(vocab_subset, 0, max_depth) for _ in range(num_rules)]
        
        all_premises = rules + facts
        premises_str = "(" + ")&(".join(all_premises) + ")"
        
        # 2. Check satisfiability
        if not is_satisfiable(premises_str):
            continue
        
        # 3. Get forced variables
        forced_vars = get_forced_variables(premises_str)
        
        # 4. Generate conclusion using intelligent strategies
        # 70% use inference-based generation, 30% random
        if random.random() < 0.7:
            candidate_conclusion = generate_conclusion_from_inference(
                premises_str, vocab_subset, forced_vars
            )
        else:
            candidate_conclusion = generate_random_expr(vocab_subset, 0, max_depth)
        
        # 5. Check if it's a valid entailment
        if not entails(premises_str, candidate_conclusion):
            continue
        
        # 6. Enhanced triviality check
        is_trivial, reason = check_triviality(
            premises_str, candidate_conclusion, all_premises, forced_vars
        )
        
        if require_nontrivial and is_trivial:
            continue
        
        # 7. Create strings
        input_str = f'{premises_str}|-'
        output_str = f'{premises_str}|-{candidate_conclusion}'
        
        # 8. Tokenize
        premise_tokenized = tokenize(input_str, seq_len)
        conclusion_tokenized = tokenize(output_str, seq_len)
        
        if premise_tokenized is None or conclusion_tokenized is None:
            continue
        
        return np.array(premise_tokenized), np.array(conclusion_tokenized), is_trivial
    
    # If we fail to generate non-trivial, allow trivial as fallback
    if require_nontrivial:
        return generate_sample(seq_len, num_vars, max_depth, require_nontrivial=False)
    
    return None, None, True


def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    # Track triviality statistics
    trivial_count = 0
    nontrivial_count = 0
    target_nontrivial = int(num_samples * config.min_nontrivial_ratio)
    
    pbar = tqdm(total=num_samples, desc=f"Generating {set_name}")
    
    while example_id < num_samples:
        # Determine if this sample should be non-trivial
        # Enforce ratio: prioritize non-trivial if below target
        require_nontrivial = (nontrivial_count < target_nontrivial) or \
                            (random.random() < config.min_nontrivial_ratio)
        
        inp, out, is_trivial = generate_sample(
            config.seq_len, 
            config.num_vars, 
            config.max_depth,
            require_nontrivial=require_nontrivial
        )
        
        if inp is None:
            continue
        
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)
        
        if is_trivial:
            trivial_count += 1
        else:
            nontrivial_count += 1
        
        pbar.update(1)
        pbar.set_postfix({
            'trivial': trivial_count,
            'non-trivial': nontrivial_count,
            'ratio': f"{nontrivial_count/(trivial_count+nontrivial_count):.2%}"
        })
    
    pbar.close()
    
    print(f"\n{set_name} Statistics:")
    print(f"  Total: {num_samples}")
    print(f"  Non-trivial: {nontrivial_count} ({nontrivial_count/num_samples:.1%})")
    print(f"  Trivial: {trivial_count} ({trivial_count/num_samples:.1%})")
    
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
        "sets": ["all"],
        "nontrivial_count": nontrivial_count,
        "trivial_count": trivial_count,
        "nontrivial_ratio": nontrivial_count / num_samples
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
