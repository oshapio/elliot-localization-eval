"""Use PyTorch SDPA for the checkpoint's NaViT attention without changing weights.

Q/K/V projections and 2-D rotary embeddings follow modeling_elliot_siglip2_navit.
Only the attention multiplication/softmax uses the memory-efficient implementation.
"""
import sys
from types import MethodType

import torch.nn.functional as F


def sdpa_forward(self, hidden_states, h_patches, w_patches, attention_mask=None,
                 output_attentions=False, spatial_shapes=None):
    if output_attentions:
        raise ValueError("SDPA does not materialize attention matrices")
    batch, length, _ = hidden_states.shape
    query, key, value = [projection(hidden_states).view(batch, length, self.num_heads, self.head_dim)
                         for projection in (self.q_proj, self.k_proj, self.v_proj)]
    source = sys.modules[type(self).__module__]
    if self.config.use_rope:
        if spatial_shapes is not None:
            cos, sin = source.get_batched_2d_rope_sin_cos(
                spatial_shapes, length, self.head_dim, base=self.config.rope_theta,
                device=hidden_states.device)
        else:
            cos, sin = source.get_2d_rope_sin_cos(
                h_patches, w_patches, self.head_dim, base=self.config.rope_theta,
                device=hidden_states.device)
        query = source.apply_rotary_pos_emb(query, cos, sin)
        key = source.apply_rotary_pos_emb(key, cos, sin)
    output = F.scaled_dot_product_attention(
        query.to(value.dtype).transpose(1, 2), key.to(value.dtype).transpose(1, 2), value.transpose(1, 2),
        attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
    return self.out_proj(output.transpose(1, 2).contiguous().view(batch, length, self.embed_dim)), None


def enable_sdpa(vision_model):
    count = 0
    for module in vision_model.modules():
        if type(module).__name__ == 'ElliotSiglip2NavitAttention':
            module.forward = MethodType(sdpa_forward, module)
            count += 1
    if count != vision_model.config.num_hidden_layers:
        raise ValueError(f"Expected one NaViT attention per layer, found {count}")
