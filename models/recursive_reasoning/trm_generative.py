from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from models.common import trunc_normal_init_
from models.layers import rms_norm, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

# --- CONFIGURAZIONE ---

class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    # Data params
    batch_size: int = 1  # Often overridden during creation
    seq_len: int
    vocab_size: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int = 0
    puzzle_emb_len: int = 16 

    # Model Architecture
    hidden_size: int
    num_heads: int
    expansion: float = 4.0
    
    # Recurrence (Universal Transformer Depth)
    H_cycles: int = 1 
    L_cycles: int = 1 
    L_layers: int = 1 

    # Tech params
    pos_encodings: str = "rope" # "rope" or "learned"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    forward_dtype: str = "bfloat16"
    
    # Flags
    mlp_t: bool = False 
    causal: bool = True  # [NEW] Fondamentale per Autoregressive

# --- MODULI ---

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.norm_eps = config.rms_norm_eps

        if self.config.mlp_t:
            # MLP-Mixer style (non raccomandato per pure autoregressive a lunghezza variabile, ma supportato)
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, 
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=config.causal # [MOD] Ora usa il flag causale
            )
            
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )

    def forward(self, cos_sin: Optional[CosSin], hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, L, D]
        
        # 1. Token Mixing (Attention or MLP-T)
        residual = hidden_states
        if self.config.mlp_t:
            # MLP-Mixer operation (transposed)
            normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
            normed = normed.transpose(1, 2)
            out = self.mlp_t(normed)
            hidden_states = residual + out.transpose(1, 2)
        else:
            # Standard Attention
            normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
            attn_out = self.self_attn(cos_sin=cos_sin, hidden_states=normed)
            hidden_states = residual + attn_out

        # 2. Channel Mixing (MLP)
        residual = hidden_states
        normed = rms_norm(hidden_states, variance_epsilon=self.norm_eps)
        mlp_out = self.mlp(normed)
        hidden_states = residual + mlp_out
        
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Universal Transformer style: input injection ad ogni step ricorrente aiuta la stabilità
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        # Embeddings
        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        
        # Puzzle Embeddings (Special handling)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # Positional Embeddings
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

        # Core Layers (Shared Weights for recurrence)
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(
            layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _ in range(self.config.L_layers)]
        )

        # State Initialization (Learnable initial state for the recursive loop)
        self.H_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1))
        self.L_init = nn.Parameter(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1))

        # Heads
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # Note: Q-head (halting) rimosso per pure autoregressive, o mantenuto opzionale se vuoi fare ACT

    def _input_embeddings(self, input_ids: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor]):
        # input_ids: [B, T]
        embedding = self.embed_tokens(input_ids.to(torch.int32))

        # Add puzzle embeddings as prefix if they exist
        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers) # [B, puzzle_dim]
            
            # Pad/Reshape to match hidden size sequence
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            
            # Prepend to sequence: [B, P_Len, D] + [B, T, D] -> [B, P+T, D]
            puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)
            embedding = torch.cat((puzzle_embedding, embedding), dim=1)

        # Learned Positional Embeddings
        if self.config.pos_encodings == "learned":
            seq_len = embedding.shape[1]
            # scale by 1/sqrt(2) to maintain forward variance
            pos_emb = self.embed_pos.embedding_weight[:seq_len].unsqueeze(0).to(self.forward_dtype)
            embedding = 0.707106781 * (embedding + pos_emb)

        return self.embed_scale * embedding

    def forward(self, input_ids: torch.Tensor, puzzle_identifiers: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 1. Embeddings
        x = self._input_embeddings(input_ids, puzzle_identifiers) # [B, SeqLen, D]
        
        # 2. Rotary Positional Info
        cos_sin = None
        if hasattr(self, "rotary_emb"):
            cos_sin = self.rotary_emb() # Precompute for max len

        # 3. Recursive Processing (Universal Transformer)
        # Invece di "pensare" nel tempo, qui applichiamo depth ricorrente.
        # z_H e z_L sono stati che si evolvono layer dopo layer.
        
        # Init states
        batch_size, seq_len, _ = x.shape
        # Broadcasting init state to batch/seq
        z_H = self.H_init.view(1, 1, -1).expand(batch_size, seq_len, -1)
        z_L = self.L_init.view(1, 1, -1).expand(batch_size, seq_len, -1)
        
        # Loop "H_cycles" times (Depth Recurrence)
        # Nota: L'attenzione è CAUSALE internamente, quindi è sicuro iterare.
        total_cycles = self.config.H_cycles
        
        for _ in range(total_cycles):
            # Passaggio attraverso il blocco L (che può avere L_cycles interni)
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(hidden_states=z_L, input_injection=z_H + x, cos_sin=cos_sin)
            
            # Aggiornamento High-level state
            z_H = self.L_level(hidden_states=z_H, input_injection=z_L, cos_sin=cos_sin)

        # 4. Output
        # Se abbiamo usato puzzle embeddings, rimuoviamoli dall'output prima della lm_head
        if self.config.puzzle_emb_ndim > 0:
            output_latent = z_H[:, self.puzzle_emb_len:]
        else:
            output_latent = z_H

        logits = self.lm_head(output_latent)
        return logits

# --- WRAPPER FINALE (Interfaccia Standard) ---

class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """
    Autoregressive Wrapper for Recursive Reasoning Model.
    Si comporta come un modello standard GPT.
    """
    def __init__(self, config_dict: dict):
        super().__init__()
        # Forzo causal=True per sicurezza
        config_dict['causal'] = True
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb if hasattr(self.inner, 'puzzle_emb') else None

    def initial_carry(self, batch):
        # Dummy method per compatibilità con vecchi script di training, se necessario.
        # In AR puro non serve "carry" tra i batch.
        return None

    def forward(self, batch: Dict[str, torch.Tensor], carry: Any = None, return_keys: List[str] = []) -> Tuple[Any, torch.Tensor, Dict, Any, bool]:
        """
        Interfaccia compatibile con il training loop che mi hai mostrato.
        
        Args:
            batch: Dizionario con 'inputs' (token ids).
            carry: Ignorato in AR training (stateless tra batch).
        """
        input_ids = batch["inputs"]
        puzzle_ids = batch.get("puzzle_identifiers", None)
        
        # 1. Forward Pass (Compute Logits for all positions)
        logits = self.inner(input_ids, puzzle_ids) # [B, T, V]
        
        # 2. Compute Loss (Next Token Prediction)
        # Shift: Input [0..T-1] predicts Target [1..T]
        # Assumiamo che batch['inputs'] contenga l'intera sequenza target inclusa.
        
        # Solitamente in training AR:
        # inputs: <BOS> A B C
        # targets: A B C <EOS>
        
        # Qui usiamo un semplice shift "in-place" su inputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous().to(torch.long)
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size), 
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # 3. Format Outputs
        # Il training loop si aspetta: carry, loss, metrics, preds, all_finish
        
        metrics = {"acc": (shift_logits.argmax(dim=-1) == shift_labels).float().mean()}
        preds = {"logits": logits} # Opzionale
        
        # Return signature: carry, loss, metrics, preds, all_finish
        return None, loss, metrics, preds, True 

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, puzzle_ids=None, temperature=1.0, top_k=None):
        """
        Funzione helper per generazione (inference).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop se superiamo la context window
            idx_cond = idx if idx.size(1) <= self.config.seq_len else idx[:, -self.config.seq_len:]
            
            # Forward
            logits = self.inner(idx_cond, puzzle_ids)
            logits = logits[:, -1, :] / temperature # Prendi solo l'ultimo step
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
