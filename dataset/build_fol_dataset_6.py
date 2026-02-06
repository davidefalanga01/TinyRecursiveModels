import os
import json
import string
import random
import numpy as np
import collections
import copy
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
PREDICATES = ['P', 'Q', 'R', 'S', 'T', 'K', 'L', 'M', 'N', 'A', 'B', 'C', 'D', 'E', 'F']

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
    seq_len: int = 256
    num_train: int = 50000
    num_test: int = 5000
    
    # FOL Specifics
    num_constants: int = 4   # Number of constants active in a sample (domain size)
    num_predicates: int = 3  # Number of predicates active
    arity: int = 2           # Max arity (1 or 2)
    
    max_rules: int = 3
    max_facts: int = 5
    
    # Curriculum: 'ground', 'unary_rules', 'binary_rules', 'mixed'
    curriculum_stage: str = "binary_rules" 
    
    seed: int = 42

class Atom:
    def __init__(self, predicate: str, args: List[str]):
        self.predicate = predicate
        self.args = args

    def __repr__(self):
        return f"{self.predicate}({','.join(self.args)})"
    
    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args
    
    def __hash__(self):
        return hash(str(self))

class Rule:
    def __init__(self, premises: List[Atom], conclusion: Atom):
        self.premises = premises
        self.conclusion = conclusion
    
    def __repr__(self):
        # P(X,Y) & Q(Y) > R(X)
        lhs = " & ".join([str(p) for p in self.premises])
        return f"{lhs} > {self.conclusion}"

# --- Logic Engine (Unification & Chaining) ---

def is_variable(term: str) -> bool:
    return term in VARIABLES

def unify(atom1: Atom, atom2: Atom, theta: Dict[str, str]) -> Optional[Dict[str, str]]:
    """
    Unify two atoms. 
    atom1 comes from the Rule (has variables).
    atom2 comes from Facts (ground truths, no variables).
    Returns extended substitution theta or None if failed.
    """
    if atom1.predicate != atom2.predicate:
        return None
    
    if len(atom1.args) != len(atom2.args):
        return None
        
    new_theta = theta.copy()
    
    for t1, t2 in zip(atom1.args, atom2.args):
        # t1 might be a variable or a constant
        # t2 is always a ground constant (from fact)
        
        if is_variable(t1):
            if t1 in new_theta:
                if new_theta[t1] != t2:
                    return None # Variable conflict: X bound to 'a' then 'b'
            else:
                new_theta[t1] = t2
        else:
            # t1 is a constant, must match exactly
            if t1 != t2:
                return None
                
    return new_theta

def subst(atom: Atom, theta: Dict[str, str]) -> Atom:
    """Apply substitution to an atom."""
    new_args = [theta.get(arg, arg) for arg in atom.args]
    return Atom(atom.predicate, new_args)

def forward_chain(facts: Set[Atom], rules: List[Rule], max_depth=6) -> Set[Atom]:
    """
    Derive all consequences from facts using rules.
    This is a simplified naive implementation.
    """
    known_facts = set(facts)
    derived = True
    
    while derived:
        derived = False
        new_facts = set()
        
        for rule in rules:
            # Try to match rule premises to known facts
            # This is a constraint satisfaction problem. 
            # For this simple dataset, simple recursion/backtracking is fine.
            
            bindings = find_bindings(rule.premises, list(known_facts), {})
            for theta in bindings:
                conclusion = subst(rule.conclusion, theta)
                if conclusion not in known_facts:
                    new_facts.add(conclusion)
                    known_facts.add(conclusion)
                    derived = True
        
    return known_facts

def find_bindings(premises: List[Atom], facts: List[Atom], theta: Dict[str, str]) -> List[Dict[str, str]]:
    """Recursive search for binding logical variables."""
    if not premises:
        return [theta]
    
    first = premises[0]
    rest = premises[1:]
    
    results = []
    
    for fact in facts:
        # Try unify first premise with this fact
        new_theta = unify(first, fact, theta)
        if new_theta is not None:
            # Continue with rest
            sub_results = find_bindings(rest, facts, new_theta)
            results.extend(sub_results)
            
    return results

# --- Generators ---

def generate_sample(config: DataProcessConfig) -> Tuple[str, str, Set[Atom]]:
    
    # 1. Select Active Vocabulary
    active_consts = CONSTANTS[:config.num_constants]
    active_preds = PREDICATES[:config.num_predicates]
    
    # 2. Generate Initial Facts
    facts = set()
    num_facts = random.randint(2, config.max_facts)
    
    for _ in range(num_facts):
        pred = random.choice(active_preds)
        # Random arity for this fact? Or consistent per predicate?
        # Let's say predicates have consistent arity for simplicity (or just random 1-2)
        # Better: Assign arity to predicates
        
        # Temp: Random arity 1 or 2
        arity = random.randint(1, config.arity)
        args = [random.choice(active_consts) for _ in range(arity)]
        facts.add(Atom(pred, args))
        
    # 3. Generate Rules based on Curriculum
    rules = []
    
    if config.curriculum_stage == "ground":
        # No rules, just facts. Derive nothing new (or identity)
        pass
    
    elif config.curriculum_stage == "unary_rules":
        # P(X) > Q(X)
        for _ in range(random.randint(1, config.max_rules)):
            p1 = random.choice(active_preds)
            p2 = random.choice(active_preds)
            if p1 == p2: continue
            
            # Unary only
            rule = Rule([Atom(p1, ['X'])], Atom(p2, ['X']))
            rules.append(rule)
            
    elif config.curriculum_stage == "binary_rules":
        # P(X,Y) > Q(Y,X) or P(X,Y) & Q(Y,Z) > R(X,Z)
        
        for _ in range(random.randint(1, config.max_rules)):
            rule_type = random.choice(['swap', 'transitive', 'match'])
            
            if rule_type == 'swap':
                # P(X,Y) > Q(Y,X)
                p1 = random.choice(active_preds)
                p2 = random.choice(active_preds)
                rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['Y', 'X'])))
                
            elif rule_type == 'transitive':
                # P(X,Y) & Q(Y,Z) > R(X,Z)
                p1 = random.choice(active_preds)
                p2 = random.choice(active_preds)
                p3 = random.choice(active_preds)
                rules.append(Rule(
                    [Atom(p1, ['X', 'Y']), Atom(p2, ['Y', 'Z'])],
                    Atom(p3, ['X', 'Z'])
                ))
            elif rule_type == 'match':
                # Identity/Map: P(X,Y) > Q(X,Y)
                p1 = random.choice(active_preds)
                p2 = random.choice(active_preds)
                rules.append(Rule([Atom(p1, ['X', 'Y'])], Atom(p2, ['X', 'Y'])))

    # 4. Derive Truth
    final_facts = forward_chain(facts, rules)
    
    # 5. Select Target
    # We want to ask if a fact is true.
    # Logic: 
    #  Input: Facts... Rules... Target: ?
    #  Output: True/False (Wait, user guide says 'Target: R(b,a)' implying generation)
    #  The user's guide example: "Target: R(b,a)".
    #  This implies we need to output ALL derived facts? Or just one?
    #  "Target: R(b,a) R(b,d) ... (30 chars)" (Line 93 in guide)
    #  So we output ALL derived facts that were NOT in the input facts?
    #  Or just all true facts?
    #  Let's follow build_fol_dataset_5.py which outputs derived facts.
    
    # Filter out initial facts to see what was DERIVED
    derived_only = final_facts - facts
    
    # If nothing derived, maybe just output 'None' or empty?
    # Or maybe we output ALL true facts?
    # Guide says: "Target: R(b,a)" from "Facts: P(a,b) > R(b,a)"
    # It implies outputting the CONSEQUENCE.
    
    # Let's output sorted list of ALL Valid Facts (Initial + Derived) or just Derived?
    # Ideally just Derived to force reasoning.
    # But if nothing derived, it's empty.
    
    # Let's go with: Output ALL valid facts (Closure).
    # Why? Because sometimes the rule is P(X)>P(X) (identity).
    # Actually, typically in these tasks we output the full state or the new/query state.
    # Guide line 34: "Metrics: Exact match... Depth-wise accuracy"
    # Let's output strict derivations.
    
    # If no rules fire, target is empty? Or "None"? 
    # Let's stick to: Output ALL TRUE FACTS.
    # That forces the model to copy initial facts + add derived ones.
    # This acts as a memory check + reasoning check.
    
    final_facts_list = sorted(list(final_facts), key=str)
    
    # Stringify
    fact_str = " ".join(sorted([str(f) for f in facts], key=str))
    rule_str = " ".join([str(r) for r in rules])
    target_str = " ".join([str(f) for f in final_facts_list])
    
    input_str = f"Facts: {fact_str} | Rules: {rule_str} | Target:"
    
    return input_str, target_str

def tokenize(text: str, seq_len: int) -> Optional[List[int]]:
    tokens = []
    # Simple tokenization by buffering
    # P(a,b) -> P, (, a, ,, b, )
    
    buffer = ""
    for char in text:
        if char == ' ':
            if buffer:
                if buffer in VOCAB: tokens.append(VOCAB[buffer])
                buffer = ""
            continue
            
        if char in SPECIALS or char in CONSTANTS or char in VARIABLES or char in PREDICATES:
            # Check if buffer was building something else (e.g. 'Facts:')
            # 'Facts:' is a single token in VOCAB? Yes.
            
            # Need strict parsing for multichar tokens like 'Facts:' vs 'F'
            # But here CONSTANTS etc are single chars.
            # SPECIALS include 'Facts:'.
            
            # Hack: Check if current char starts a multi-char token?
            # 'Facts:' starts with 'F'. PREDICATES has 'F' too?
            # PREDICATES = ['P'...'F']
            
            buffer += char
            
            # Check if buffer is a valid token
            if buffer in VOCAB:
                tokens.append(VOCAB[buffer])
                buffer = ""
            elif buffer in ["Facts", "Rules", "Target"]:
                # Wait for the colon
                pass
            elif len(buffer) == 1 and buffer in VOCAB:
                # Immediate token (like '(', ')')
                # But wait, 'P' is a token, but 'Facts:' starts with 'F' (if F is a predicate?)
                # To avoid ambiguity, let's process word by word.
                pass
                
    # Re-do tokenization safely
    # Split by space first?
    # "Facts: P(a,b)" -> ["Facts:", "P(a,b)"]
    
    raw_tokens = []
    # Custom split considering punctuation
    curr = ""
    for char in text:
        if char == ' ':
            if curr: raw_tokens.append(curr)
            curr = ""
        elif char in ['(', ')', ',', '>', '&', '|']:
            if curr: raw_tokens.append(curr)
            raw_tokens.append(char)
            curr = ""
        else:
            curr += char
    if curr: raw_tokens.append(curr)
    
    final_ids = []
    for t in raw_tokens:
        if t in VOCAB:
            final_ids.append(VOCAB[t])
        else:
            # Might be 'Facts:' which is in VOCAB?
            # Yes, SPECIALS has 'Facts:'
            # But if t is 'P' and P is in VOCAB? Yes.
            pass
            
    # Better tokenizer approach:
    # 1. Split by ' ' to get chunks like "Facts:", "P(a,b)", "|", ">"
    # 2. For chunks like "P(a,b)", split further into P, (, a, ,, b, )
    
    tokens = []
    chunks = text.split(' ')
    for chunk in chunks:
        if not chunk: continue
        
        # Check standard keywords
        if chunk in VOCAB:
            tokens.append(VOCAB[chunk])
            continue
            
        # Parse P(a,b) or a,b or >
        i = 0
        while i < len(chunk):
            # Try 1 char
            c = chunk[i]
            if c in VOCAB:
                 tokens.append(VOCAB[c])
                 i += 1
            else:
                # Should not happen in this controlled gen
                print(f"Warning: Unknown char {c}")
                i += 1
                
    if len(tokens) > seq_len - 1: return None
    tokens.append(VOCAB['end'])
    tokens = tokens + [VOCAB['pad']] * (seq_len - len(tokens))
    return tokens


# --- Standard Boilerplate ---
def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int):
    results = {k: [] for k in ["inputs", "labels", "puzzle_indices", "group_indices", "puzzle_identifiers"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0); results["group_indices"].append(0)
    
    valid_count = 0
    with tqdm(total=num_samples, desc=f"Generating {set_name}") as pbar:
        while valid_count < num_samples:
            inp_str, targ_str = generate_sample(config)
            full_text = f"{inp_str} {targ_str}"
            
            input_tokens = tokenize(inp_str, config.seq_len)
            label_tokens = tokenize(full_text, config.seq_len)
            
            if (input_tokens is None) or (label_tokens is None): continue
                
            results["inputs"].append(np.array(input_tokens))
            results["labels"].append(np.array(label_tokens))
            example_id += 1; puzzle_id += 1
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(valid_count)
            results["group_indices"].append(puzzle_id)
            valid_count += 1
            pbar.update(1)

    final_results = {
        "inputs": np.stack(results["inputs"]),
        "labels": np.stack(results["labels"]),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    for k, v in final_results.items(): np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
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
    random.seed(config.seed); np.random.seed(config.seed)
    print(f"Generating FOL ({config.curriculum_stage}) in: {config.output_dir}")
    convert_subset("train", config, config.num_train)
    convert_subset("test", config, config.num_test)

if __name__ == "__main__":
    cli()
