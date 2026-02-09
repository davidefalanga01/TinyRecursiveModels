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


def create_target_mask(labels: torch.Tensor, target_token_id: int = 30) -> torch.Tensor:
    """
    Creates a mask that is True only for tokens AFTER "Target:" token.
    
    Args:
        labels: [batch_size, seq_len] tensor of token ids
        target_token_id: The token id for "Target:" (default: 30)
    
    Returns:
        mask: [batch_size, seq_len] boolean tensor
              True for tokens after "Target:", False elsewhere
    """
    batch_size, seq_len = labels.shape
    mask = torch.zeros_like(labels, dtype=torch.bool)
    
    # Find "Target:" token position for each sequence in batch
    for batch_idx in range(batch_size):
        # Find all positions where "Target:" appears
        target_positions = (labels[batch_idx] == target_token_id).nonzero(as_tuple=True)[0]
        
        if len(target_positions) > 0:
            # Take the first occurrence (should only be one)
            target_pos = target_positions[0].item()
            # Activate mask for all positions AFTER "Target:"
            # We want to predict the token immediately after "Target:"
            mask[batch_idx, target_pos + 1:] = True
    
    # Also mask out padding (token 0) and any other ignore tokens
    mask = mask & (labels != IGNORE_LABEL_ID)
    
    return mask


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, use_target_masking: bool = True, target_token_id: int = 30):
        """
        Args:
            model: The model to wrap
            loss_type: Loss function name ('stablemax_cross_entropy' or 'softmax_cross_entropy')
            use_target_masking: If True, compute loss only on tokens after "Target:"
                               If False, compute loss on entire sequence (original behavior)
            target_token_id: The token id for "Target:" (default: 30)
        """
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.use_target_masking = use_target_masking
        self.target_token_id = target_token_id
        
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
            # ==========================================
            # MASKED LOSS SETUP (NEW)
            # ==========================================
            if self.use_target_masking:
                # Create mask that's True only after "Target:" token
                target_mask = create_target_mask(labels, target_token_id=self.target_token_id)
            else:
                # Original behavior: mask only padding
                target_mask = (labels != IGNORE_LABEL_ID)
            
            # This is the mask used for LOSS computation
            loss_mask = target_mask
            
            # ==========================================
            # METRICS (keep original full-sequence mask for metrics)
            # ==========================================
            # For metrics, we want to evaluate the entire sequence
            full_mask = (labels != IGNORE_LABEL_ID)
            
            # Preds
            preds = torch.argmax(outputs["logits"], dim=-1)
            outputs["preds"] = preds

            # Correctness (on full sequence for compatibility)
            full_loss_counts = full_mask.sum(-1)
            full_loss_divisor = full_loss_counts.clamp_min(1).unsqueeze(-1)

            is_correct = full_mask & (preds == labels)
            seq_is_correct = is_correct.sum(-1) == full_loss_counts
            
            # --- SET ACCURACY (unchanged) ---
            batch_size = labels.shape[0]
            set_matches = []
            
            cpu_labels = labels.detach().cpu().numpy()
            cpu_preds = preds.detach().cpu().numpy()
            cpu_mask = full_mask.detach().cpu().numpy()
            
            for b in range(batch_size):
                lab_seq = cpu_labels[b]
                pred_seq = cpu_preds[b]
                
                # 1. Extract Facts from Label
                facts_set = set()
                f_indices = np.where(lab_seq == 28)[0]  # Facts: token
                if len(f_indices) > 0:
                    f_start = f_indices[0] + 1
                    delims = np.where((lab_seq == 31) | (lab_seq == 29))[0]  # | or Rules:
                    valid_delims = delims[delims > f_start]
                    if len(valid_delims) > 0:
                        f_end = valid_delims[0]
                        facts_slice = lab_seq[f_start:f_end]
                        facts_set = {x for x in facts_slice if 2 <= x <= 27}

                # 2. Extract Target and Pred
                l_slice = lab_seq
                p_slice = pred_seq
                
                t_indices = np.where(lab_seq == 30)[0]  # Target: token
                if len(t_indices) > 0:
                    t_idx = t_indices[0]
                    l_slice = lab_seq[t_idx+1:]
                    p_slice = pred_seq[t_idx+1:]
                
                l_set = {x for x in l_slice if 2 <= x <= 27}
                p_set = {x for x in p_slice if 2 <= x <= 27}

                # 3. Check for correctness
                missing = l_set - p_set
                hallucinated = p_set - l_set - facts_set
                
                is_valid = (len(missing) == 0) and (len(hallucinated) == 0)
                set_matches.append(is_valid)
                
            set_accuracy_tensor = torch.tensor(set_matches, device=labels.device, dtype=torch.float32)

            # Metrics (using full sequence counts for compatibility)
            valid_metrics = new_carry.halted & (full_loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / full_loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),
                
                # Exact set accuracy (for FOL)
                "set_accuracy": (valid_metrics & set_accuracy_tensor.bool()).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }
            
            # Add diagnostic metric to track masking
            if self.use_target_masking:
                target_token_counts = loss_mask.sum(-1)
                metrics["target_tokens"] = torch.where(valid_metrics, target_token_counts, 0).sum()

        # ==========================================
        # LOSS COMPUTATION (MODIFIED)
        # ==========================================
        # Use the target_mask for loss (only tokens after "Target:")
        loss_counts = loss_mask.sum(-1)
        loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)
        
        # Compute loss only on masked tokens
        lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=loss_mask) / loss_divisor).sum()
        
        # Q-halt loss (unchanged)
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