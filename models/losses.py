from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np

IGNORE_LABEL_ID = 0


def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = 0, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = 0):
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        
    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # --- NEW METRIC: SET ACCURACY ---
            # Used to evaluate logical correctness ignoring order of atoms.
            # Compatible with both FOL (atoms with parens) and Logic Chain (variables).
            
            batch_size = labels.shape[0]
            set_matches = []
            
            # Move to CPU for set operations
            cpu_labels = labels.detach().cpu().numpy()
            cpu_preds = preds.detach().cpu().numpy()
            
            # Known Token IDs across datasets
            # FOL (Dataset 6): Facts: 43, Target: 45, |: 51, ): 49, Rules: 44
            # Logic (Dataset 2/3): Facts: 28, Target: 30, |: 31/33, Rules: 29
            TRGT_IDS = [45, 30]
            FACT_IDS = [43, 28]
            
            # Helper: Find first matching ID in sequence
            def find_first(seq, candidates):
                for cand in candidates:
                    indices = np.where(seq == cand)[0]
                    if len(indices) > 0:
                        return cand, indices[0]
                return None, None

            def extract_atoms_fol(seq):
                # FOL: Atoms end with ')' (49)
                atoms = set()
                current_atom = []
                for token in seq:
                    if token == 0: continue # Pad
                    if token == 1: break    # End
                    # Reset on known delimiters or new section headers
                    if token in [43, 44, 45, 51]: 
                        current_atom = []
                        continue
                    current_atom.append(token)
                    if token == 49: # ')'
                        atoms.add(tuple(current_atom))
                        current_atom = []
                return atoms

            def extract_atoms_simple(seq):
                # Logic Chain: Atoms are single tokens (variables A-Z, 2..27)
                # Ignore special tokens >= 28
                atoms = set()
                for token in seq:
                    if token == 0: continue
                    if token == 1: break
                    if token >= 28: continue # Ignore headers/delims if present
                    atoms.add((token,))
                return atoms
            
            for b in range(batch_size):
                lab_seq = cpu_labels[b]
                pred_seq = cpu_preds[b]
                
                # 1. Detect Format based on Target token
                trgt_id, t_idx = find_first(lab_seq, TRGT_IDS)
                
                # Default slices (full sequence if no Target found)
                l_slice = lab_seq
                p_slice = pred_seq
                
                if t_idx is not None:
                    l_slice = lab_seq[t_idx+1:]
                    p_slice = pred_seq[t_idx+1:]
                
                # Determine mode: check if slice has ')' (49)
                # FOL mode if 49 is present, else Simple mode
                is_fol = (49 in l_slice)
                
                extract_fn = extract_atoms_fol if is_fol else extract_atoms_simple
                
                l_set = extract_fn(l_slice)
                p_set = extract_fn(p_slice)
                
                # 2. Extract Facts (for hallucination check)
                facts_set = set()
                fact_id, f_idx = find_first(lab_seq, FACT_IDS)
                
                if f_idx is not None:
                    f_start = f_idx + 1
                    # Find end of facts: | or Rules: or Target:
                    # Delims: 51(FOL |), 31(Chain |), 33(Branch |), 44(FOL Rules), 29(Chain Rules)
                    # We just look for the next token that looks like a delimiter
                    # Simple heuristic: scan until we hit a known delimiter ID
                    delims = [51, 31, 33, 44, 29, 45, 30] # Includes Target too just in case
                    
                    f_slice = []
                    for k in range(f_start, len(lab_seq)):
                        tok = lab_seq[k]
                        if tok in delims:
                            break
                        f_slice.append(tok)
                    
                    facts_set = extract_fn(np.array(f_slice))

                # 3. Check for correctness
                missing = l_set - p_set
                hallucinated = p_set - l_set - facts_set
                
                is_valid = (len(missing) == 0) and (len(hallucinated) == 0)
                set_matches.append(is_valid)
                
            set_accuracy_tensor = torch.tensor(set_matches, device=labels.device, dtype=torch.float32)

            # Metrics
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                
                #exact set accuracy (for fol)
                "set_accuracy": (valid_metrics & set_accuracy_tensor.bool()).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()
            
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()