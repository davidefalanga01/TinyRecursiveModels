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
            # Used to evaluate logical correctness ignoring order
            # This is expensive for huge batches, but fine for Tiny Models
            # We must ignore padding (0) or specialized ignore indices
            
            # 1. Extract sets from labels and preds using the mask
            # Note: This runs on CPU usually because of list comps, or we can use tensor ops if shape is fixed.
            # Given variable length masking, list comprehension is safer/easier to implement quickly.
            batch_size = labels.shape[0]
            set_matches = []
            
            # Move to CPU for set operations to avoid cuda sync overhead loop
            cpu_labels = labels.detach().cpu().numpy()
            cpu_preds = preds.detach().cpu().numpy()
            cpu_mask = mask.detach().cpu().numpy()
            
            for b in range(batch_size):
                # Filter out ignored tokens and padding (usually 0)
                # But FIRST, find the anchor "Target:" (30) in the LABEL to slice both
                
                lab_seq = cpu_labels[b]
                pred_seq = cpu_preds[b]
                
                # 1. Extract Facts from Label (Facts: ... |)
                # Token 28=Facts:, 31=|, 29=Rules:
                facts_set = set()
                f_indices = np.where(lab_seq == 28)[0]
                if len(f_indices) > 0:
                    f_start = f_indices[0] + 1
                    # Look for end of facts section (either | or Rules:)
                    delims = np.where((lab_seq == 31) | (lab_seq == 29))[0]
                    valid_delims = delims[delims > f_start]
                    if len(valid_delims) > 0:
                        f_end = valid_delims[0]
                        facts_slice = lab_seq[f_start:f_end]
                        # Facts are variables (2-27)
                        facts_set = {x for x in facts_slice if 2 <= x <= 27}

                # 2. Extract Target and Pred
                # Default to full sequence if Target not found (fallback)
                l_slice = lab_seq
                p_slice = pred_seq
                
                # Check for Target token (30) in LABEL
                t_indices = np.where(lab_seq == 30)[0]
                if len(t_indices) > 0:
                    t_idx = t_indices[0]
                    # We want everything AFTER the target token
                    l_slice = lab_seq[t_idx+1:]
                    p_slice = pred_seq[t_idx+1:]
                
                # Now build sets, filtering for Variables (2-27)
                # This automatically handles padding (0) and special tokens (>=28)
                l_set = {x for x in l_slice if 2 <= x <= 27}
                p_set = {x for x in p_slice if 2 <= x <= 27}

                # 3. Check for correctness
                # A. All target symbols must be predicted (Missing = Empty)
                missing = l_set - p_set
                
                # B. No hallucinations allowed UNLESS they are in the input Facts
                # (Hallucinated = Pred - Target - Facts = Empty)
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