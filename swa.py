import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.cache_utils import Cache


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class SlidingWindowAttention(nn.Module):

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.keep_window_size = config.keep_window_size

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.hidden_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.hidden_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.hidden_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.hidden_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_mask = self.prepare_sliding_window_mask(
            hidden_states,
            keep_window_size=self.keep_window_size,
            attention_mask=attention_mask,
        )
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attention_mask=attn_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None
    
    def prepare_sliding_window_mask(
        self,
        hidden_states: torch.Tensor,
        keep_window_size: int = 2048,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """

        Args:
            hidden_states (`torch.Tensor`): The input hidden_states, used to determine the minimum value of the current input precision.
            keep_window_size (`int`): The window size of tokens that are not masked.
            attention_mask (`torch.Tensor`, *optional*): attention mask of shape `(batch_size, 1, query_sequence_length, key_sequence_length)`.
        """
        batch_size, q_len = hidden_states.shape[:2]
        device = hidden_states.device
        dtype = hidden_states.dtype
        
        if attention_mask is None:
            k_len = q_len
        else:
            k_len = attention_mask.shape[-1]
        
        if k_len <= keep_window_size:
            return attention_mask

        min_value = torch.finfo(dtype).min
    
        query_indices = torch.arange(q_len, device=device).unsqueeze(1)  # [q_len, 1]
        key_indices = torch.arange(k_len, device=device).unsqueeze(0)    # [1, k_len]

        half_window = keep_window_size // 2
        last_visible_idx = torch.clamp(query_indices + half_window, max=k_len-1)
        first_visible_idx = torch.clamp(query_indices - half_window, min=0)
        key_visible = torch.logical_and(
            key_indices >= first_visible_idx,
            key_indices <= last_visible_idx
        )
        sliding_mask = torch.where(key_visible, torch.tensor(0.0, device=device, dtype=dtype), torch.tensor(min_value, device=device, dtype=dtype))
        
        # [batch_size, 1, q_len, k_len]
        sliding_mask = sliding_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, q_len, k_len)

        return sliding_mask