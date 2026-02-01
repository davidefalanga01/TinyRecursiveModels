import os
import json
import string
import random
import itertools
import numpy as np
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm


# Vocabulary for Propositional Logic
VARS = list(string.ascii_lowercase)
OPS = ['&', '|', '>']

# IMPROVED VOCABULARY - Added truth value tokens
VOCAB = {
    'pad': 0,
    **{v: i+1 for i, v in enumerate(VARS)},
    '&': 27, '|': 28, '>': 29, '~': 30,
    '(': 31, ')': 32, '|-': 33,
    'T': 34,   # TRUE
    'F': 35,   # FALSE
    '?': 36,   # UNKNOWN/UNDETERMINED
    'end': 37
}
INV_VOCAB = {v: k for k, v in VOCAB.items()}


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/logic_improved"
    seq_len: int = 96  # Increased to accommodate new format
    num_train: int = 10000
    num_test: int = 1000
    num_vars: int = 4
    max_depth: int = 2
    subsample_size: Optional[int] = None
    seed: int = 42
    min_inference_steps: int = 1  # Minimum reasoning steps required
    use_curriculum: bool = True  # Enable curriculum learning
    difficulty_distribution: Tuple[float, float, float] = (0.3, 0.4, 0.3)  # easy, medium, hard


def eval_logic_expr(expr_str, env):
    """
    Safely evaluate a logical expression given variable assignments.
    Returns None if evaluation fails.
    """
    try:
        # Replace logical operators with Python equivalents
        py_expr = (expr_str.replace('&', ' and ')
                          .replace('|', ' or ')
                          .replace('>', ' <= ')  # a > b means "a implies b" = "not a or b"
                          .replace('~', ' not '))
        
        # Note: Python's <= is used because a>b in logic means "if a then b"
        # which is equivalent to "not a or b", but for material implication
        # we need to handle it differently
        py_expr = py_expr.replace(' <= ', ' IMPLIES ')
        
        def implies(a, b):
            return (not a) or b
        
        local_env = {'IMPLIES': implies, 'not': lambda x: not x}
        return eval(py_expr, {}, {**env, **local_env})
    except Exception as e:
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


def count_inference_steps(premises_str, forced_vars, vocab_subset):
    """
    Estimate the number of inference steps needed to derive forced variables.
    
    A variable requires inference if:
    1. It's not directly stated as a fact in premises
    2. It requires combining multiple rules
    
    Returns: number of non-trivial inference steps
    """
    # Extract direct facts (single variables appearing as atomic propositions)
    direct_facts = set()
    
    # Simple heuristic: check if variable appears standalone in premises
    # This is a simplified check - you might want to improve this
    for var in vocab_subset:
        # Check patterns like "a&", "&a", "(a)", "a)" indicating direct assertion
        if f'({var})' in premises_str or f'&{var}' in premises_str or f'{var}&' in premises_str:
            # Could be a direct fact
            # Verify it's not negated
            if f'~({var})' not in premises_str and f'~{var}' not in premises_str:
                direct_facts.add(var)
    
    # Count variables that were forced but not directly stated
    derived_vars = set(forced_vars.keys()) - direct_facts
    
    # Additional complexity: check for implication chains
    # If we have a>b and b>c and a is forced, then c requires 2 steps
    inference_steps = len(derived_vars)
    
    # Bonus complexity for variables that are forced to False (requires contradiction reasoning)
    false_forced = sum(1 for v, val in forced_vars.items() if val is False)
    inference_steps += false_forced * 0.5  # Weight false inferences slightly higher
    
    return int(inference_steps)


def tokenize(text, seq_len=96):
    """
    Tokenize text to integer sequence.
    Now handles T, F, ? tokens for truth values.
    """
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
        # Skip spaces (they're not in vocab)
    
    if len(clean_tokens) > seq_len - 1:
        return None
    
    clean_tokens.append(VOCAB['end'])
    tokens = clean_tokens + [VOCAB['pad']] * (seq_len - len(clean_tokens))
    return tokens


def generate_random_expr(vocab_subset, depth=0, max_depth=2, force_var_usage=None):
    """
    Generate a random logical expression.
    
    Args:
        vocab_subset: List of variables to use
        depth: Current recursion depth
        max_depth: Maximum depth to recurse
        force_var_usage: Set of variables that must be used (for coherent problems)
    """
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        # Base case: return a variable
        if force_var_usage and random.random() < 0.7:
            return random.choice(list(force_var_usage))
        return random.choice(vocab_subset)
    
    op = random.choice(OPS + ['~'])
    if op == '~':
        sub_expr = generate_random_expr(vocab_subset, depth+1, max_depth, force_var_usage)
        return f"~({sub_expr})"
    else:
        left = generate_random_expr(vocab_subset, depth+1, max_depth, force_var_usage)
        right = generate_random_expr(vocab_subset, depth+1, max_depth, force_var_usage)
        return f"({left}{op}{right})"


def generate_structured_premises(vocab_subset, difficulty='medium'):
    """
    Generate premises with controlled difficulty and guaranteed inference opportunities.
    
    Difficulty levels:
    - easy: 2 vars, 1-2 simple rules, depth 1, direct implications
    - medium: 3 vars, 2-3 rules, depth 1-2, some chaining
    - hard: 4 vars, 2-4 rules, depth 2, complex chaining and negations
    
    Returns: premises_str, expected_forced_vars_count
    """
    
    if difficulty == 'easy':
        num_vars = min(2, len(vocab_subset))
        selected_vars = vocab_subset[:num_vars]
        max_depth = 1
        
        # Pattern: Simple modus ponens (a, a>b => b)
        a, b = selected_vars[0], selected_vars[1] if num_vars > 1 else selected_vars[0]
        
        patterns = [
            # Pattern 1: Direct implication
            ([a, f'({a}>{b})'], 2),
            # Pattern 2: Conjunction
            ([f'({a}&{b})'], 2),
            # Pattern 3: Double implication
            ([f'({a}>{b})', f'({b}>{a})', a], 2),
        ]
        
        premises, expected = random.choice(patterns)
        
    elif difficulty == 'medium':
        num_vars = min(3, len(vocab_subset))
        selected_vars = vocab_subset[:num_vars]
        max_depth = 2
        
        a, b, c = selected_vars[0], selected_vars[1], selected_vars[2] if num_vars > 2 else selected_vars[0]
        
        patterns = [
            # Pattern 1: Implication chain (a, a>b, b>c => a,b,c)
            ([a, f'({a}>{b})', f'({b}>{c})'], 3),
            # Pattern 2: Disjunction with negation
            ([f'({a}|{b})', f'~({a})'], 1),
            # Pattern 3: Complex conjunction
            ([f'(({a}>{b})&({b}>{c}))', a], 3),
            # Pattern 4: Mixed operators
            ([f'({a}&({b}|{c}))', f'~({c})'], 2),
        ]
        
        premises, expected = random.choice(patterns)
        
    else:  # hard
        num_vars = min(4, len(vocab_subset))
        selected_vars = vocab_subset[:num_vars]
        max_depth = 2
        
        # Use all variables for harder problems
        vars_sample = random.sample(selected_vars, min(4, len(selected_vars)))
        
        # Generate complex nested structures
        num_rules = random.randint(2, 4)
        premises = []
        
        # Ensure at least one fact to ground the reasoning
        premises.append(random.choice(vars_sample))
        
        # Add complex rules
        for _ in range(num_rules - 1):
            premises.append(generate_random_expr(vars_sample, 0, max_depth, set(vars_sample[:2])))
        
        expected = -1  # Unknown for random generation
    
    premises_str = '&'.join([f'({p})' if not p.startswith('(') else p for p in premises])
    
    return premises_str, expected


def generate_sample(seq_len=96, vocab_subset=None, difficulty='medium', min_inference_steps=1):
    """
    Generate a single FOL sample with IMPROVED FORMAT.
    
    NEW FORMAT:
    Input:  "premises |-"
    Output: "a T b F c ? d T"  (truth value for each variable in fixed order)
    
    This eliminates:
    - Need to copy input in output
    - Order ambiguity (fixed variable order)
    - Conjunction formatting issues
    
    Args:
        seq_len: Maximum sequence length
        vocab_subset: Variables to use (e.g., ['a', 'b', 'c', 'd'])
        difficulty: 'easy', 'medium', or 'hard'
        min_inference_steps: Minimum number of non-trivial inferences required
    
    Returns:
        input_tokens, output_tokens, metadata_dict
    """
    if vocab_subset is None:
        vocab_subset = VARS[:4]
    
    max_attempts = 100
    for attempt in range(max_attempts):
        # 1. Generate structured premises based on difficulty
        premises_str, expected_forced = generate_structured_premises(vocab_subset, difficulty)
        
        # 2. Check satisfiability
        if not is_satisfiable(premises_str):
            continue
        
        # 3. Compute forced variables
        forced_vars = get_forced_variables(premises_str)
        
        # 4. Check if sample meets minimum inference requirements
        inference_steps = count_inference_steps(premises_str, forced_vars, vocab_subset)
        
        if inference_steps < min_inference_steps:
            continue
        
        # 5. Build IMPROVED output format: "var1 T/F/? var2 T/F/? ..."
        # Use FIXED order (alphabetically sorted) for consistency
        output_parts = []
        for var in sorted(vocab_subset):
            if var in forced_vars:
                truth_val = 'T' if forced_vars[var] else 'F'
            else:
                truth_val = '?'
            output_parts.extend([var, truth_val])
        
        # 6. Create input and output strings
        input_str = f'{premises_str}|-'
        output_str = ''.join(output_parts)  # e.g., "aTbFc?dT"
        
        # 7. Tokenize
        input_tokens = tokenize(input_str, seq_len)
        output_tokens = tokenize(output_str, seq_len)
        
        if input_tokens is None or output_tokens is None:
            continue
        
        # 8. Metadata for analysis
        metadata = {
            'difficulty': difficulty,
            'num_forced': len(forced_vars),
            'inference_steps': inference_steps,
            'premises_length': len(premises_str),
            'num_vars_used': len(get_vars_in_expr(premises_str))
        }
        
        return np.array(input_tokens), np.array(output_tokens), metadata
    
    # Failed to generate valid sample
    return None, None, None


def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    """
    Generate dataset with curriculum learning support.
    """
    np.random.seed(config.seed)
    random.seed(config.seed)
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    # Statistics tracking
    stats = {
        'easy': 0,
        'medium': 0,
        'hard': 0,
        'total_inference_steps': 0,
        'total_forced_vars': 0
    }
    
    # Difficulty distribution
    if config.use_curriculum and set_name == 'train':
        # For training: use curriculum (more easy early on)
        # Split dataset into thirds
        third = num_samples // 3
        difficulty_schedule = (
            ['easy'] * third + 
            ['medium'] * third + 
            ['hard'] * (num_samples - 2 * third)
        )
        random.shuffle(difficulty_schedule)  # Shuffle to avoid strict ordering
    else:
        # For test: use configured distribution
        easy_pct, medium_pct, hard_pct = config.difficulty_distribution
        difficulty_schedule = (
            ['easy'] * int(num_samples * easy_pct) +
            ['medium'] * int(num_samples * medium_pct) +
            ['hard'] * int(num_samples * hard_pct)
        )
        # Fill remaining
        while len(difficulty_schedule) < num_samples:
            difficulty_schedule.append(random.choice(['easy', 'medium', 'hard']))
        random.shuffle(difficulty_schedule)
    
    vocab_subset = VARS[:config.num_vars]
    
    pbar = tqdm(total=num_samples, desc=f"Generating {set_name}")
    
    attempt_count = 0
    max_total_attempts = num_samples * 10
    
    while example_id < num_samples and attempt_count < max_total_attempts:
        attempt_count += 1
        
        difficulty = difficulty_schedule[example_id] if example_id < len(difficulty_schedule) else 'medium'
        
        inp, out, metadata = generate_sample(
            config.seq_len,
            vocab_subset,
            difficulty=difficulty,
            min_inference_steps=config.min_inference_steps
        )
        
        if inp is None:
            continue
        
        # Successfully generated sample
        results["inputs"].append(inp)
        results["labels"].append(out)
        example_id += 1
        puzzle_id += 1
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(puzzle_id)
        
        # Update statistics
        stats[difficulty] += 1
        stats['total_inference_steps'] += metadata['inference_steps']
        stats['total_forced_vars'] += metadata['num_forced']
        
        pbar.update(1)
        pbar.set_postfix({
            'easy': stats['easy'],
            'med': stats['medium'],
            'hard': stats['hard'],
            'avg_infer': f"{stats['total_inference_steps']/example_id:.2f}"
        })
    
    pbar.close()
    
    if example_id < num_samples:
        print(f"\nWARNING: Only generated {example_id}/{num_samples} samples after {attempt_count} attempts")
    
    print(f"\n{set_name} Statistics:")
    print(f"  Total: {example_id}")
    print(f"  Easy: {stats['easy']} ({stats['easy']/example_id:.1%})")
    print(f"  Medium: {stats['medium']} ({stats['medium']/example_id:.1%})")
    print(f"  Hard: {stats['hard']} ({stats['hard']/example_id:.1%})")
    print(f"  Avg inference steps: {stats['total_inference_steps']/example_id:.2f}")
    print(f"  Avg forced vars: {stats['total_forced_vars']/example_id:.2f}")
    
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
        "mean_puzzle_examples": example_id / puzzle_id if puzzle_id > 0 else 0,
        "total_puzzles": puzzle_id,
        "sets": ["all"],
        "difficulty_stats": stats,
        "vocab": VOCAB,
        "inv_vocab": INV_VOCAB
    }
    
    # Save dataset
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"\nDataset saved to {save_dir}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Generate train and test datasets."""
    print("=" * 60)
    print("IMPROVED FOL Dataset Generation")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Sequence length: {config.seq_len}")
    print(f"  Num variables: {config.num_vars}")
    print(f"  Max depth: {config.max_depth}")
    print(f"  Min inference steps: {config.min_inference_steps}")
    print(f"  Curriculum learning: {config.use_curriculum}")
    print(f"  Difficulty distribution: Easy={config.difficulty_distribution[0]}, "
          f"Med={config.difficulty_distribution[1]}, Hard={config.difficulty_distribution[2]}")
    print("=" * 60)
    print()
    
    convert_subset("train", config, config.num_train)
    print()
    convert_subset("test", config, config.num_test)
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    cli()