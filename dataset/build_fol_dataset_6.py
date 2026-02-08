
import os
import json
import string
import random
import numpy as np
import collections
from typing import Optional, List, Tuple, Set, Dict
from pydantic import BaseModel
from argdantic import ArgParser
from tqdm import tqdm

# --- Vocabulary ---
# Constants: a-z
CONSTANTS = list(string.ascii_lowercase)
# Variables: X, Y, Z, U, V, W
VARIABLES = ['X', 'Y', 'Z', 'U', 'V', 'W']
# Predicates: P, Q, R, S, T, ...
PREDICATES = ['P', 'Q', 'R', 'S', 'T', 'K', 'L', 'M', 'N']

# Structure: Facts, Rules, Target, Logic symbols
SPECIALS = ['Facts:', 'Rules:', 'Target:', '>', '&', '(', ')', ',', '|']

VOCAB = {
    'pad': 0,
    'end': 1,
    **{c: i + 2 for i, c in enumerate(CONSTANTS)},
    **{v: i + 2 + len(CONSTANTS) for i, v in enumerate(VARIABLES)},
    **{p: i + 2 + len(CONSTANTS) + len(VARIABLES) for i, p in enumerate(PREDICATES)},
    **{s: i + 2 + len(CONSTANTS) + len(VARIABLES) + len(PREDICATES) for i, s in enumerate(SPECIALS)}
}

cli = ArgParser()

class DataProcessConfig(BaseModel):
    output_dir: str = "data/fol_basic"
    seq_len: int = 160  # Increased for more complex FOL formulas
    num_train: int = 20000
    num_test: int = 2000
    
    # FOL Specifics
    num_constants: int = 4   # Number of constants active in a sample (domain size)
    num_predicates: int = 3  # Number of predicates active
    
    min_rules: int = 1
    max_rules: int = 3
    min_facts: int = 2
    max_facts: int = 5
    
    # Curriculum: 'ground', 'unary_rules', 'binary_rules', 'mixed'
    curriculum_stage: str = "binary_rules" 
    
    # Distractor rules (rules that won't fire)
    num_distractors: int = 2
    
    seed: int = 42

class Atom:
    """Represents a logical atom like P(a,b) or Q(x)"""
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate
        self.args = tuple(args)  # Use tuple for immutability and hashing
        self.arity = len(args)

    def __repr__(self):
        return f"{self.predicate}({','.join(self.args)})"
    
    def __eq__(self, other):
        return isinstance(other, Atom) and self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        return hash((self.predicate, self.args))
    
    def is_ground(self) -> bool:
        """Check if all arguments are constants (no variables)"""
        return all(not is_variable(arg) for arg in self.args)

class Rule:
    """Represents a logic rule like P(X,Y) & Q(Y,Z) > R(X,Z)"""
    def __init__(self, premises: List[Atom], conclusion: Atom):
        self.premises = premises
        self.conclusion = conclusion
    
    def __repr__(self):
        if not self.premises:
            return f"> {self.conclusion}"
        lhs = "&".join([str(p) for p in self.premises])
        return f"{lhs}>{self.conclusion}"
    
    def get_variables(self) -> Set[str]:
        """Get all variables used in this rule"""
        vars_set = set()
        for premise in self.premises:
            vars_set.update(arg for arg in premise.args if is_variable(arg))
        vars_set.update(arg for arg in self.conclusion.args if is_variable(arg))
        return vars_set

# --- Logic Engine (Unification & Forward Chaining) ---

def is_variable(term: str) -> bool:
    """Check if a term is a logical variable"""
    return term in VARIABLES

def unify(atom1: Atom, atom2: Atom, theta: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Unify two atoms with existing substitution theta.
    atom1: Pattern from rule (may contain variables)
    atom2: Ground fact (no variables)
    Returns: Extended substitution or None if unification fails
    """
    if atom1.predicate != atom2.predicate:
        return None
    
    if len(atom1.args) != len(atom2.args):
        return None
        
    new_theta = theta.copy()
    
    for t1, t2 in zip(atom1.args, atom2.args):
        if is_variable(t1):
            # Variable in pattern
            if t1 in new_theta:
                # Variable already bound, must match
                if new_theta[t1] != t2:
                    return None
            else:
                # Bind variable to constant
                new_theta[t1] = t2
        else:
            # Constant in pattern, must match exactly
            if t1 != t2:
                return None
                
    return new_theta

def subst(atom: Atom, theta: Dict[str, str]) -> Atom:
    """Apply substitution theta to an atom"""
    new_args = [theta.get(arg, arg) for arg in atom.args]
    return Atom(atom.predicate, new_args)

def forward_chain_with_depth(facts: Set[Atom], rules: List[Rule], max_depth: int = 10) -> Tuple[Set[Atom], Dict[Atom, int]]:
    """
    Derive all consequences from facts using rules.
    Returns: (all_facts, depth_map) where depth_map tracks derivation depth
    """
    known_facts = set(facts)
    depth_map = {fact: 0 for fact in facts}  # Initial facts have depth 0
    
    current_depth = 0
    changed = True
    
    while changed and current_depth < max_depth:
        changed = False
        current_depth += 1
        
        # Store new facts derived in this iteration
        new_facts_this_iteration = []
        
        for rule in rules:
            # Find all bindings that satisfy the rule premises
            bindings = find_bindings(rule.premises, list(known_facts), {})
            
            for theta in bindings:
                # Apply substitution to conclusion
                conclusion = subst(rule.conclusion, theta)
                
                if conclusion not in known_facts and conclusion.is_ground():
                    new_facts_this_iteration.append(conclusion)
                    changed = True
        
        # Add all new facts with the current depth
        for fact in new_facts_this_iteration:
            known_facts.add(fact)
            depth_map[fact] = current_depth
        
    return known_facts, depth_map

def find_bindings(premises: List[Atom], facts: List[Atom], theta: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Recursively find all variable bindings that satisfy the premises.
    Uses backtracking search.
    """
    if not premises:
        return [theta]
    
    first = premises[0]
    rest = premises[1:]
    
    results = []
    
    for fact in facts:
        # Try to unify first premise with this fact
        new_theta = unify(first, fact, theta)
        if new_theta is not None:
            # Recursively find bindings for remaining premises
            sub_results = find_bindings(rest, facts, new_theta)
            results.extend(sub_results)
            
    return results

# --- Predicate Arity Assignment ---

def assign_predicate_arities(predicates: List[str], curriculum_stage: str) -> Dict[str, int]:
    """
    Assign consistent arities to predicates based on curriculum.
    This ensures P(x) is always unary and Q(x,y) is always binary.
    """
    arity_map = {}
    
    if curriculum_stage == "ground":
        # Mix of unary and binary
        for pred in predicates:
            arity_map[pred] = random.choice([1, 2])
    elif curriculum_stage == "unary_rules":
        # All unary predicates
        for pred in predicates:
            arity_map[pred] = 1
    elif curriculum_stage == "binary_rules":
        # All binary predicates
        for pred in predicates:
            arity_map[pred] = 2
    elif curriculum_stage == "mixed":
        # Mix of unary and binary
        for i, pred in enumerate(predicates):
            arity_map[pred] = 1 if i % 2 == 0 else 2
    
    return arity_map

# --- Generators ---

def generate_sample(config: DataProcessConfig) -> Tuple[str, str]:
    """Generate a single FOL reasoning sample"""
    
    # 1. Select Active Vocabulary
    active_consts = CONSTANTS[:config.num_constants]
    active_preds = PREDICATES[:config.num_predicates]
    
    # 2. Assign consistent arities to predicates
    arity_map = assign_predicate_arities(active_preds, config.curriculum_stage)
    
    # 3. Generate Initial Facts
    facts = set()
    num_facts = random.randint(config.min_facts, config.max_facts)
    
    attempts = 0
    while len(facts) < num_facts and attempts < 100:
        attempts += 1
        pred = random.choice(active_preds)
        arity = arity_map[pred]
        args = [random.choice(active_consts) for _ in range(arity)]
        facts.add(Atom(pred, args))
    
    # 4. Generate Rules based on Curriculum
    rules = []
    num_rules = random.randint(config.min_rules, config.max_rules)
    
    if config.curriculum_stage == "ground":
        # No rules with variables, just facts
        pass
    
    elif config.curriculum_stage == "unary_rules":
        # P(X) > Q(X)
        for _ in range(num_rules):
            if len(active_preds) < 2:
                break
            p1, p2 = random.sample(active_preds, 2)
            rule = Rule([Atom(p1, ['X'])], Atom(p2, ['X']))
            rules.append(rule)
            
    elif config.curriculum_stage == "binary_rules":
        # Various binary rule patterns
        for _ in range(num_rules):
            rule_type = random.choice(['swap', 'transitive', 'projection', 'match'])
            
            if rule_type == 'swap':
                # P(X,Y) > Q(Y,X)
                if len(active_preds) < 2:
                    continue
                p1, p2 = random.sample(active_preds, 2)
                rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['Y', 'X'])))
                
            elif rule_type == 'transitive':
                # P(X,Y) & Q(Y,Z) > R(X,Z)
                if len(active_preds) < 2:
                    continue
                preds = random.sample(active_preds, min(3, len(active_preds)))
                if len(preds) == 2:
                    preds.append(preds[0])  # Reuse if not enough
                p1, p2, p3 = preds[0], preds[1], preds[2]
                rules.append(Rule(
                    [Atom(p1, ['X', 'Y']), Atom(p2, ['Y', 'Z'])],
                    Atom(p3, ['X', 'Z'])
                ))
                
            elif rule_type == 'projection':
                # P(X,Y) > Q(X,X) or P(X,Y) > Q(Y,Y)
                if len(active_preds) < 2:
                    continue
                p1, p2 = random.sample(active_preds, 2)
                var = random.choice(['X', 'Y'])
                rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, [var, var])))
                
            elif rule_type == 'match':
                # P(X,Y) > Q(X,Y)
                if len(active_preds) < 2:
                    continue
                p1, p2 = random.sample(active_preds, 2)
                rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['X', 'Y'])))
                
    elif config.curriculum_stage == "mixed":
        # Mix of unary and binary rules
        for _ in range(num_rules):
            available_unary = [p for p in active_preds if arity_map[p] == 1]
            available_binary = [p for p in active_preds if arity_map[p] == 2]
            
            if available_unary and random.random() < 0.5:
                # Unary rule
                if len(available_unary) >= 2:
                    p1, p2 = random.sample(available_unary, 2)
                    rules.append(Rule([Atom(p1, ['X'])], Atom(p2, ['X'])))
            elif available_binary:
                # Binary rule
                rule_type = random.choice(['swap', 'transitive', 'match'])
                if rule_type == 'swap' and len(available_binary) >= 2:
                    p1, p2 = random.sample(available_binary, 2)
                    rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['Y', 'X'])))
                elif rule_type == 'transitive' and len(available_binary) >= 2:
                    preds = random.sample(available_binary, min(3, len(available_binary)))
                    if len(preds) == 2:
                        preds.append(preds[0])
                    p1, p2, p3 = preds[0], preds[1], preds[2]
                    rules.append(Rule(
                        [Atom(p1, ['X', 'Y']), Atom(p2, ['Y', 'Z'])],
                        Atom(p3, ['X', 'Z'])
                    ))
                elif rule_type == 'match' and len(available_binary) >= 2:
                    p1, p2 = random.sample(available_binary, 2)
                    rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['X', 'Y'])))

    # 5. Generate Distractor Rules (rules that won't fire)
    distractor_rules = []
    attempts = 0
    while len(distractor_rules) < config.num_distractors and attempts < 50:
        attempts += 1
        
        if config.curriculum_stage == "unary_rules":
            # Create rule that references non-existent predicate in premise
            available = [p for p in active_preds if arity_map[p] == 1]
            if len(available) >= 2:
                # Pick a predicate that doesn't appear in facts
                fact_preds = {f.predicate for f in facts}
                unused_preds = [p for p in available if p not in fact_preds]
                if unused_preds:
                    p1 = random.choice(unused_preds)
                    p2 = random.choice(available)
                    distractor_rules.append(Rule([Atom(p1, ['X'])], Atom(p2, ['X'])))
                    
        elif config.curriculum_stage == "binary_rules":
            # Create rule with premise that won't match any facts
            available = [p for p in active_preds if arity_map[p] == 2]
            if len(available) >= 2:
                fact_preds = {f.predicate for f in facts}
                unused_preds = [p for p in available if p not in fact_preds]
                if unused_preds:
                    p1 = random.choice(unused_preds)
                    p2 = random.choice(available)
                    distractor_rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['X', 'Y'])))
    
    # 6. Derive all consequences
    final_facts, depth_map = forward_chain_with_depth(facts, rules)
    
    # 7. Build output strings
    # Sort facts for consistency
    initial_facts_list = sorted(list(facts), key=str)
    all_rules = rules + distractor_rules
    random.shuffle(all_rules)  # Randomize rule order to make task harder
    
    # Format strings
    fact_str = " ".join([str(f) for f in initial_facts_list])
    rule_str = " ".join([str(r) for r in all_rules]) if all_rules else ""
    
    # Target: Output only DERIVED facts (not initial facts), sorted by depth then alphabetically
    derived_facts = final_facts - facts
    
    # Sort by depth, then by string representation
    derived_sorted = sorted(list(derived_facts), key=lambda f: (depth_map[f], str(f)))
    target_str = " ".join([str(f) for f in derived_sorted])
    
    # Build input/output
    if rule_str:
        input_str = f"Facts: {fact_str} | Rules: {rule_str} | Target:"
    else:
        input_str = f"Facts: {fact_str} | Rules: | Target:"
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    """
    Tokenize FOL text into vocabulary indices.
    Handles complex tokens like 'Facts:' and 'P(a,b)'
    """
    tokens = []
    
    # Split by spaces first
    chunks = text.split(' ')
    
    for chunk in chunks:
        if not chunk:
            continue
            
        # Check if whole chunk is a vocab token (e.g., 'Facts:', 'Rules:', '|')
        if chunk in VOCAB:
            tokens.append(VOCAB[chunk])
            continue
        
        # Otherwise, parse character by character
        i = 0
        while i < len(chunk):
            # Try multi-character tokens first (Facts:, Rules:, Target:)
            found = False
            for length in [7, 6, 5, 4, 3, 2]:  # Try longer matches first
                if i + length <= len(chunk):
                    substr = chunk[i:i+length]
                    if substr in VOCAB:
                        tokens.append(VOCAB[substr])
                        i += length
                        found = True
                        break
            
            if found:
                continue
            
            # Single character
            c = chunk[i]
            if c in VOCAB:
                tokens.append(VOCAB[c])
                i += 1
            else:
                # Unknown character - this shouldn't happen with controlled generation
                print(f"Warning: Unknown character '{c}' in chunk '{chunk}'")
                return None
    
    # Check length
    if len(tokens) > seq_len - 1:
        return None
    
    # Add end token and padding
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    
    return tokens

# --- Standard Boilerplate ---

def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    """Generate and save a dataset subset"""
    results = {
        k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]
    }
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    valid_count = 0
    with tqdm(total=num_samples, desc=f"Generating {set_name}") as pbar:
        while valid_count < num_samples:
            try:
                inp_str, targ_str = generate_sample(config)
                full_text = f"{inp_str} {targ_str}"
                
                input_tokens = tokenize(inp_str, config.seq_len)
                label_tokens = tokenize(full_text, config.seq_len)
                
                if input_tokens is None or label_tokens is None:
                    continue
                    
                results["inputs"].append(np.array(input_tokens))
                results["labels"].append(np.array(label_tokens))
                example_id += 1
                puzzle_id += 1
                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(valid_count)
                results["group_indices"].append(puzzle_id)
                valid_count += 1
                pbar.update(1)
                
            except Exception as e:
                print(f"Error generating sample: {e}")
                continue

    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    for k, v in final_results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump({
            "seq_len": config.seq_len,
            "vocab_size": len(VOCAB),
            "pad_id": VOCAB['pad'],
            "ignore_label_id": 0,
            "blank_identifier_id": 0,
            "total_groups": len(final_results["group_indices"]) - 1,
            "mean_puzzle_examples": 1.0,
            "num_puzzle_identifiers": valid_count,
            "total_puzzles": valid_count,
            "sets": ["all"]
        }, f)
    
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump([f"fol_{i}" for i in range(valid_count)], f)
    
    with open(os.path.join(config.output_dir, "vocab.json"), "w") as f:
        json.dump(VOCAB, f)

@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Main entry point for data generation"""
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    print(f"Generating FOL Dataset ({config.curriculum_stage})")
    print(f"Output: {config.output_dir}")
    print(f"Vocabulary size: {len(VOCAB)}")
    print(f"Sequence length: {config.seq_len}")
    
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)
    
    print("Done!")

if __name__ == "__main__":
    cli()
