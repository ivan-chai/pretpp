import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.functional import *
from torch.nn.functional import _canonical_mask, _none_or_dtype
from typing import Optional
from hotpp.data import PaddedBatch
from hotpp.nn import SimpleTransformer
from hotpp.nn.encoder.transformer.rope import MultiheadAttentionRoPE, multi_head_attention_rope_forward

from .history_token_strategy import add_token_to_the_end


def replace_with_lf_attention(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_with_lf_attention(module)
        if isinstance(module, torch.nn.MultiheadAttention):
            setattr(model, name, MultiheadAttentionLongFormer(module))


class LongFormer(SimpleTransformer):
    """An extension of the transformer model with extra <memory-tokens> and recurrent inference.

    Args:
        chunk_size: The maximum size of input chunk at each iteration.
        kernel_size: Convolution kernel size or None to disable convolution mask.
        global_frequency: Frequency of the global token.
    """
    def __init__(self, input_size, *, kernel_size=None, global_frequency=0.1, **kwargs):
        if kwargs.pop("causal", True):
            raise ValueError("LongFormer can't be causal.")
        super().__init__(input_size, **kwargs)
        replace_with_lf_attention(self)
        self.kernel_size = kernel_size
        self.global_frequency = global_frequency

    def _select_global(self, seq_lens, embedding=False):
        device = seq_lens.device
        max_length = seq_lens.max().item()
        assert max_length > 0
        max_tokens = max(1, int(round(self.global_frequency * max_length)))
        if not embedding:
            # Use random locations at train to prevent overfitting.
            global_positions = 1 + torch.randperm(max_length - 1, device=device)[:max_tokens - 1].sort()[0]  # (R - 1) in [1, L), sorted.
            global_positions = torch.cat([torch.zeros([1], device=device, dtype=torch.long), global_positions])  # (R) in [0, L), sorted.
        else:
            # Use regular locations at inference for preproducibility.
            global_positions = torch.arange(0, max_length, max(1, int(round(max_length / max_tokens))), device=device)
        return global_positions

    def _make_attention_mask(self, embeddings, global_positions, embedding=False):
        """Make attention mask.

        NOTE:
        0: standard attention.
        1: no attention.
        -1: global attention.
        """
        batch_size, length = embeddings.shape
        # Put ones at diagonals.
        mask = torch.ones(length, length, dtype=torch.long, device=embeddings.device)
        if self.kernel_size:
            mask = torch.tril(conv_mask, self.kernel_size)
            mask = torch.triu(conv_mask, -self.kernel_size)

        # Add causality to all tokens except global.
        mask = 1 - torch.tril(mask)

        # Put ones at global rows and cols.
        mask[:, global_positions] = 0
        mask[global_positions] = -1

        # Make last tokens global.
        mask = mask[None].repeat(batch_size, 1, 1)  # (B, L, L).
        last = (embeddings.seq_lens - 1).clip(min=0)  # (B).
        mask.scatter_(2, last[:, None, None].expand(batch_size, length, 1), 0)
        mask.scatter_(1, last[:, None, None].expand(batch_size, 1, length), -1)

        return mask  # Zeros for allowed interractions.

    def transform(self, embeddings, embedding=False):
        global_positions = self._select_global(embeddings.seq_lens, embedding=embedding)
        attention_mask = self._make_attention_mask(embeddings, global_positions,
                                                   embedding=embedding)
        if attention_mask.ndim == 3:
            if attention_mask.shape[0] == len(embeddings):
                attention_mask = attention_mask.repeat_interleave(self.n_head, dim=0)
        outputs = self.encoder(embeddings.payload,
                               mask=attention_mask.float(),  # To skip "canonical" checks.
                               src_key_padding_mask=~embeddings.seq_len_mask.bool(),
                               is_causal=False,
                               rope=self.rope)  # (B, L, D).
        return PaddedBatch(outputs, embeddings.seq_lens)

    def embed(self, x, timestamps):
        embeddings = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).
        embeddings = PaddedBatch(self.positional(embeddings.payload, timestamps.payload), embeddings.seq_lens)  # (B, L, D).
        if self.rope is not None:
            with self.rope.cache(timestamps.payload):
                outputs = self.transform(embeddings, embedding=True)
        else:
            outputs = self.transform(embeddings, embedding=True)
        embeddings = outputs.payload.take_along_dim((outputs.seq_lens[:, None, None] - 1).clip(min=0), 1).squeeze(1)  # (B, D).
        return embeddings

    def forward(self, x, timestamps, states=None, return_states=False):
        if return_states:
            raise NotImplementedError("RecurrentMemoryTransformer doesn't support states return.")
        embeddings = PaddedBatch(self.input_projection(x.payload), x.seq_lens)  # (B, L, D).
        embeddings = PaddedBatch(self.positional(embeddings.payload, timestamps.payload), embeddings.seq_lens)  # (B, L, D).
        if self.rope is not None:
            with self.rope.cache(timestamps.payload):
                outputs = self.transform(embeddings)
        else:
            outputs = self.transform(embeddings)
        states = None
        return outputs, states


class MultiheadAttentionLongFormer(MultiheadAttentionRoPE):
    def __init__(self, attn):
        p = next(iter(attn.parameters()))
        factory_kwargs = {"device": p.device, "dtype": p.dtype}
        super().__init__(embed_dim=attn.embed_dim,
                         num_heads=attn.num_heads,
                         group_size=attn.group_size,
                         dropout=attn.dropout,
                         bias=attn.in_proj_bias is not None,
                         add_bias_kv=attn.bias_k is not None,
                         add_zero_attn=attn.add_zero_attn,
                         kdim=attn.kdim,
                         vdim=attn.vdim,
                         batch_first=attn.batch_first,
                         **factory_kwargs)
        self.load_state_dict(attn.state_dict())
        embed_dim = attn.embed_dim
        if not self._qkv_same_embed_dim:
            self.g_q_proj_weight = Parameter(self.q_proj_weight.clone())
            self.g_k_proj_weight = Parameter(self.k_proj_weight.clone())
            self.g_v_proj_weight = Parameter(self.v_proj_weight.clone())
            self.register_parameter("g_in_proj_weight", None)
        else:
            self.g_in_proj_weight = Parameter(self.in_proj_weight.clone())
            self.register_parameter("g_q_proj_weight", None)
            self.register_parameter("g_k_proj_weight", None)
            self.register_parameter("g_v_proj_weight", None)

    def _reset_parameters(self):
        super()._reset_parameters()
        if not self._qkv_same_embed_dim:
            if hasattr(self, "g_q_proj_weight"):
                # Already initialized.
                self.g_q_proj_weight.copy_(self.q_proj_weight)
                self.g_k_proj_weight.copy_(self.k_proj_weight)
                self.g_v_proj_weight.copy_(self.v_proj_weight)
        else:
            if hasattr(self, "g_in_proj_weight"):
                # Already initialized.
                self.g_in_proj_weight.copy_(self.in_proj_weight)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
        rope=None
    ) -> tuple[Tensor, Optional[Tensor]]:
        rope = rope if rope is not None else getattr(self, "_rope", [None])[0]
        if attn_mask is not None:
            if not torch.is_floating_point(attn_mask):
                raise ValueError("Expected Floating mask")
        is_global = attn_mask[..., 0] == -1 if attn_mask is not None else None  # (L) or (BH, L).
        if (is_global is not None) and (is_global.ndim == 2):
            is_global = is_global[::self.num_heads]  # (B, L).
        attn_mask = attn_mask > 0 if attn_mask is not None else None  # (L, L) or (B, L, L).
        n_global = is_global.long().sum().item() if is_global is not None else 0
        if (rope is None) and (self.group_size == 1) and (n_global == 0):
            return super().forward(query, key, value,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=need_weights,
                                   attn_mask=attn_mask,
                                   average_attn_weights=average_attn_weights,
                                   is_causal=is_causal)
        if query.dim() != 3:
            if self.batch_first:
                raise ValueError("Expected batched input with shape (B, L, D).")
            else:
                raise ValueError("Expected batched input with shape (L, B, D).")

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, (
            "MultiheadAttention does not support NestedTensor outside of its fast path. "
        )

        if self.batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # QKV: (L, B, D).

        attn_output, attn_output_weights = multi_head_attention_rope_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            self.in_proj_weight,
            self.in_proj_bias,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=not self._qkv_same_embed_dim,
            q_proj_weight=self.q_proj_weight,
            k_proj_weight=self.k_proj_weight,
            v_proj_weight=self.v_proj_weight,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
            rope=rope,
            enable_gqa=self.group_size > 1
        )
        # attn_output: (L, B, D).
        # attn_output_weights: (B, L, L).
        if n_global > 0:
            l, b, d = attn_output.shape
            g_attn_output, g_attn_output_weights = multi_head_attention_rope_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.g_in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=not self._qkv_same_embed_dim,
                q_proj_weight=self.g_q_proj_weight,
                k_proj_weight=self.g_k_proj_weight,
                v_proj_weight=self.g_v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                rope=rope,
                enable_gqa=self.group_size > 1
            )
            if is_global.ndim == 1:
                is_global = is_global[None, :]  # (B, L).
            attn_output = torch.where(is_global.T.unsqueeze(-1), g_attn_output, attn_output)
            if attn_output_weights is not None:
                attn_output_weights = torch.where(is_global.unsqueeze(-1), g_attn_output_weights, attn_output_weights)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
