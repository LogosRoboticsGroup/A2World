# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from collections import defaultdict, namedtuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.multiview_dit import MultiViewDiT
from imaginaire.utils.graph import create_cuda_graph
from cosmos_predict2.models.text2image_dit import (
    Attention,
    Block,
    SACConfig,
    VideoRopePosition3DEmb,
)

from megatron.core import parallel_state
from torch.distributed import ProcessGroup, get_process_group_ranks
from torchvision import transforms

VideoSize = namedtuple("VideoSize", ["T", "H", "W"])

@contextmanager
def temporarily_remove_hooks(root: torch.nn.Module):
    saved = []  # [(m, f_pre_dict, f_dict, b_dict)]
    for m in root.modules():
        fpre = dict(m._forward_pre_hooks)
        fwd  = dict(m._forward_hooks)
        bwd  = dict(m._backward_hooks)
        if fpre or fwd or bwd:
            saved.append((m, fpre, fwd, bwd))
            m._forward_pre_hooks.clear()
            m._forward_hooks.clear()
            m._backward_hooks.clear()
    try:
        yield
    finally:
        for m, fpre, fwd, bwd in saved:
            for _, fn in fpre.items():
                m.register_forward_pre_hook(fn)
            for _, fn in fwd.items():
                m.register_forward_hook(fn)
            for _, fn in bwd.items():
                m.register_full_backward_hook(fn)

class LatentToCrossAttEmb(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        mid_channels: int = 256,
        out_dim: int = 1024,
        pool_t: int = 16,
        pool_h: int = 12,
        pool_w: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_dim = out_dim
        self.pool_t = pool_t
        self.pool_h = pool_h
        self.pool_w = pool_w

        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool3d((pool_t, pool_h, pool_w))
        self.token_norm = nn.LayerNorm(out_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, H, W)
        return: (B, 1536, out_dim)
        """
        B, C, T, H, W = x.shape
        assert C == self.in_channels, f"Expected C={self.in_channels}, got {C}"

        x = self.conv_in(x)
        x = self.pool(x)

        B, C_out, T_p, H_p, W_p = x.shape
        assert T_p * H_p * W_p == 1536, f"Got {T_p*H_p*W_p} tokens, expect 1536"
        x = x.view(B, C_out, T_p * H_p * W_p)          # (B, C_out, 1536)
        x = x.permute(0, 2, 1).contiguous()           # (B, 1536, C_out)

        x = self.token_norm(x)
        x = x + self.token_mlp(x)

        return x  # (B, 1536, out_dim)

class FramePackMemory(nn.Module):
    def __init__(self, in_ch: int, D: int,
                 kernels=((1,2,2),(1,4,4),(1,8,8)),
                 lam: float = 2.0,
                 max_tokens: int = 1024,
                 tail: str = "one"):
        super().__init__()
        self.lam, self.max_tokens, self.tail = lam, max_tokens, tail
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool3d(k, k),
                nn.Conv3d(in_ch, D, kernel_size=1)
            ) for k in kernels
        ])

    def _levels(self, T: int):
        L = len(self.proj)
        return [min(int(round(math.log(max(1, i+1), self.lam))), L-1) for i in range(T)]

    def forward(self, hist_B_C_T_H_W: torch.Tensor) -> torch.Tensor | None:
        if hist_B_C_T_H_W is None: return None
        B,C,T,H,W = hist_B_C_T_H_W.shape
        if T == 0: return None

        levels = self._levels(T)
        toks = []
        for lvl, block in enumerate(self.proj):
            sel = [t for t,l in enumerate(levels) if l == lvl]
            if not sel: continue
            x = hist_B_C_T_H_W[:, :, sel]                   # [B,C,Tk,H,W]
            x = block(x)                                    # [B,D,Tk,H',W']
            res = x.flatten(2).transpose(1,2)
            toks.append(res)        # → [B, Tk*H'*W', D]

        if not toks:
            return None

        mem = torch.cat(toks, dim=1)                        # [B, L_hist, D]
        mem = mem[:, :self.max_tokens]
        return mem

def sinusoidal_pe(T: int, D: int, device):
    dim = torch.arange(D, device=device).float()
    pos = torch.arange(T, device=device).float()[:, None]
    div = torch.exp(-torch.arange(0, D, 2, device=device).float() * (torch.log(torch.tensor(10000.0)) / D))
    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(torch.bfloat16)

class ZeroConv3d1x1(nn.Conv3d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.weight); nn.init.zeros_(self.bias)

class ActionLatentConditioner(nn.Module):
    """
    blacks: (B, Cz=16, tl, hl, wl)  →  cond_D: (B, tl, D),  cond_3D: (B, tl, 3D)
    """
    def __init__(
        self,
        in_channels: int,          # 16
        model_channels: int,       # D
        nheads: int = 8,
        nlayers: int = 2,
        ff_mult: int = 4,
        use_temporal_transformer: bool = False,
        zero_init_gate: float = 0.0,
        norm: str = "ln",
    ):
        super().__init__()
        D = model_channels
        self.D = D
        self.use_temporal_transformer = use_temporal_transformer

        self.proj_in = ZeroConv3d1x1(in_channels, D)
        self.dw_spatial = nn.Conv3d(D, D, kernel_size=(1,3,3), padding=(0,1,1), groups=D, bias=False)
        self.pw_spatial = nn.Conv3d(D, D, kernel_size=1, bias=False)

        if use_temporal_transformer:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=D, nhead=nheads, dim_feedforward=ff_mult*D,
                batch_first=True, dropout=0.0, activation="gelu", norm_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.to_D   = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, D))
        self.to_3D  = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 3*D))

    @torch.no_grad()
    def _ensure_time_match(self, x_B_C_T_H_W, T_target: int):
        B, C, T, H, W = x_B_C_T_H_W.shape
        if T == T_target:
            return x_B_C_T_H_W
        x = x_B_C_T_H_W.permute(0,1,3,4,2).contiguous()      # (B,C,H,W,T)
        x = F.interpolate(x, size=T_target, mode="linear", align_corners=False)  # 1D over T
        x = x.permute(0,1,4,2,3).contiguous()                # (B,C,T,H,W)
        return x

    def forward(
        self,
        blacks: torch.Tensor,                                   # (B,16,tl,hl,wl)
        T_target: int | None = None,                            # 与 t_embedding 的 T 对齐
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,  # (B,1,tl,hl,wl)
    ):
        assert blacks.dim() == 5, f"expect (B,Cz,T,H,W), got {tuple(blacks.shape)}"
        B, Cz, T, H, W = blacks.shape
        if T_target is not None:
            blacks = self._ensure_time_match(blacks, T_target)
            T = T_target

        x = blacks                                                    # (B,16,T,H,W)
        x = self.proj_in(x)
        x = self.dw_spatial(x); x = self.pw_spatial(x)                # (B,D,T,H,W)

        if condition_video_input_mask_B_C_T_H_W is not None:
            gen_mask = (1 - condition_video_input_mask_B_C_T_H_W).to(dtype=x.dtype, device=x.device)  # (B,1,T,H,W)
            x = x * gen_mask

        x = x.mean(dim=(3,4))                                         # (B,D,T)
        x = rearrange(x, 'b d t -> b t d').contiguous()               # (B,T,D)

        pe = sinusoidal_pe(T, self.D, x.device)                       # (T,D)
        x = x + pe

        if self.use_temporal_transformer:
            x = self.temporal_encoder(x)                              # (B,T,D)
        else:
            pass

        cond_D  = self.to_D(x)                         # (B,T,D)
        cond_3D = self.to_3D(x)                        # (B,T,3D)
        return cond_D, cond_3D


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbed(nn.Module):
    """
    PatchEmbed is a module for embedding patches from an input tensor by applying either 3D or 2D convolutional layers,
    depending on the . This module can process inputs with temporal (video) and spatial (image) dimensions,
    making it suitable for video and image processing tasks. It supports dividing the input into patches
    and embedding each patch into a vector of size `out_channels`.

    Parameters:
    - spatial_patch_size (int): The size of each spatial patch.
    - temporal_patch_size (int): The size of each temporal patch.
    - in_channels (int): Number of input channels. Default: 3.
    - out_channels (int): The dimension of the embedding vector for each patch. Default: 768.
    - bias (bool): If True, adds a learnable bias to the output of the convolutional layers. Default: True.
    """

    def __init__(
        self,
        spatial_patch_size: int,
        temporal_patch_size: int,
        in_channels: int = 3,
        out_channels: int = 768,
    ):
        super().__init__()
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.proj = nn.Sequential(
            Rearrange(
                "b c (t r) (h m) (w n) -> b t h w (c r m n)",
                r=temporal_patch_size,
                m=spatial_patch_size,
                n=spatial_patch_size,
            ),
            nn.Linear(
                in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size, out_channels, bias=False
            ),
        )
        self.dim = in_channels * spatial_patch_size * spatial_patch_size * temporal_patch_size

        self.init_weights()

    def init_weights(self) -> None:
        std = 1.0 / math.sqrt(self.dim)
        torch.nn.init.trunc_normal_(self.proj[1].weight, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchEmbed module.

        Parameters:
        - x (torch.Tensor): The input tensor of shape (B, C, T, H, W) where
            B is the batch size,
            C is the number of channels,
            T is the temporal dimension,
            H is the height, and
            W is the width of the input.

        Returns:
        - torch.Tensor: The embedded patches as a tensor, with shape b t h w c.
        """
        assert x.dim() == 5
        _, _, T, H, W = x.shape
        assert H % self.spatial_patch_size == 0 and W % self.spatial_patch_size == 0, (
            f"H,W {(H, W)} should be divisible by spatial_patch_size {self.spatial_patch_size}"
        )
        assert T % self.temporal_patch_size == 0
        x = self.proj(x)
        return x

class MultiViewCrossAttention(Attention):
    def __init__(self, *args, state_t: int = None, **kwargs) -> None:  # noqa: RUF013
        super().__init__(*args, **kwargs)
        assert self.qkv_format == "bshd", "MultiViewCrossAttention only supports qkv_format='bshd'"
        self.state_t = state_t

    def forward(self, x, context=None, rope_emb=None):
        assert not self.is_selfattn, "MultiViewCrossAttention does not support self-attention"
        B, L, D = x.shape
        n_cameras = context.shape[1] // 512
        x_B_L_D = rearrange(x, "B (V L) D -> (V B) L D", V=n_cameras)
        context_B_M_D = rearrange(context, "B (V M) D -> (V B) M D", V=n_cameras) if context is not None else None
        x_B_L_D = super().forward(x_B_L_D, context_B_M_D, rope_emb=rope_emb)
        x_B_L_D = rearrange(x_B_L_D, "(V B) L D -> B (V L) D", V=n_cameras)
        return x_B_L_D

class CrossViewAttention(Attention):
    def __init__(self, *args, cross_view_attn_map: Dict[int, List[int]], **kwargs):
        super().__init__(*args, **kwargs)
        del self.attn_op
        if self.backend == "transformer_engine":
            from transformer_engine.pytorch.attention import DotProductAttention

            self.attn_op = DotProductAttention(
                self.n_heads,
                self.head_dim,
                num_gqa_groups=self.n_heads,
                attention_dropout=0,
                qkv_format=self.qkv_format,
                attn_mask_type="padding",  # important
                attention_type="cross",  # important
            )
        else:
            raise NotImplementedError(f"Backend {self.backend} not supported")
        self.cross_view_attn_map = cross_view_attn_map
        self.max_neighbors = max(len(neighbors) for neighbors in cross_view_attn_map.values())
        self.neighbor_indices = None
        self.neighbor_mask = None

    def forward(self, x, view_indices_B_V, sv_video_size: VideoSize):
        """
        x: (B, V, L, D)
        view_indices_B_V: (B, V)
        sv_video_size: VideoSize (T, H, W), where T * H * W = L
        """
        assert not self.is_selfattn, "CrossViewAttention does not support self-attention"
        B, V, L, D = x.shape
        T, H, W = sv_video_size
        assert T * H * W == L, f"T * H * W != L: {T * H * W} != {L}"

        # move time dimension to batch dimension
        x = rearrange(x, "b v (t h w) d -> (b t) v (h w) d", t=T, h=H, w=W)
        B, V, L, D = x.shape

        view_indices_B_V = view_indices_B_V.repeat_interleave(T, dim=0).long()

        # Create neighbor indices and mask on the fly, only once.
        if self.neighbor_indices is None or self.neighbor_indices.device != x.device:
            num_total_views = len(self.cross_view_attn_map)
            neighbor_indices = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.long, device=x.device)
            neighbor_mask = torch.zeros((num_total_views, self.max_neighbors), dtype=torch.bool, device=x.device)
            for i in range(num_total_views):
                neighbors = self.cross_view_attn_map[i]
                for j, neighbor_idx in enumerate(neighbors):
                    neighbor_indices[i, j] = neighbor_idx
                    neighbor_mask[i, j] = True
            self.neighbor_indices = neighbor_indices
            self.neighbor_mask = neighbor_mask

        num_total_views = len(self.cross_view_attn_map)
        view_indices_to_tensor_pos = torch.full(
            (B, num_total_views), -1, dtype=torch.long, device=x.device
        )  # include out of range view index
        b_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, V).long()
        view_indices_to_tensor_pos[b_indices, view_indices_B_V] = (
            torch.arange(V, device=x.device).unsqueeze(0).expand(B, -1)
        )

        neighbor_view_indices = self.neighbor_indices[view_indices_B_V]  # may include out of range view index
        gather_tensor_pos = view_indices_to_tensor_pos[
            b_indices.unsqueeze(2), neighbor_view_indices
        ]  # [B, V, max_neighbors], out of range view index will be -1

        # Sort to move all -1 to the end, which is convenient for creating attention mask.
        gather_tensor_pos, sorted_indices = torch.sort(gather_tensor_pos, dim=-1, descending=True)

        b_indices_for_gather = torch.arange(B, device=x.device)[:, None, None]
        # Clamp to avoid index error. Masked values will be ignored in attention.
        neighbor_features = x[
            b_indices_for_gather, torch.clamp(gather_tensor_pos, min=0)
        ]  # [B, V, max_neighbors, L, C]

        # Prepare for attention
        query = self.q_proj(rearrange(x, "b v l c -> (b v) l c"))  # [B*V, L, C]
        context = rearrange(neighbor_features, "b v n l c -> (b v) (n l) c")  # [B*V, max_neighbors*L, C]
        key = self.k_proj(context)
        value = self.v_proj(context)

        q, k, v = map(
            lambda t: rearrange(t, "b ... (h d) -> b ... h d", h=self.n_heads, d=self.head_dim),
            (query, key, value),
        )

        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Create attention mask
        is_neighbor_present = gather_tensor_pos != -1  # [B, V, max_neighbors]
        mask_for_input_views = self.neighbor_mask[view_indices_B_V]  # [B, V, n]

        # Reorder mask_for_input_views to match the sorted gather_tensor_pos
        mask_for_input_views = torch.gather(mask_for_input_views, -1, sorted_indices)
        final_mask = is_neighbor_present & mask_for_input_views

        mask_per_view = rearrange(final_mask, "b v n -> (b v) n")  # [BV, n]
        mask_kv = mask_per_view.repeat_interleave(L, dim=1)  # [BV, n*L]

        # Reshape mask to [batch_size, 1, 1, max_seqlen_kv] as per official documentation.
        mask = rearrange(mask_kv, "bv l_kv -> bv 1 1 l_kv")  # [BV, 1, 1, n*L]
        atten_mask_kv = ~mask  # 0 means keep, 1 means mask
        atten_mask_q = torch.zeros(query.shape[0], 1, 1, query.shape[1]).to(atten_mask_kv)

        attention_output = self.attn_op(q, k, v, attention_mask=(atten_mask_q, atten_mask_kv))
        attention_output = attention_output.flatten(2)  # [B*V, L, H*D]
        output = self.output_dropout(self.output_proj(attention_output))
        output = rearrange(output, "(b v) l d -> b v l d", v=V)
        # recover time dimension from batch to seq
        output = rearrange(output, "(b t) v (h w) d -> b v (t h w) d", t=T, h=H, w=W)
        return output

    def set_context_parallel_group(self, process_group, ranks, stream):
        raise NotImplementedError("Cross View Attention doesn't need communication")


class MultiViewCrossBlock(Block):
    """
    A transformer block that takes n_cameras as input. This block
    """

    def __init__(
        self,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
        self_attention_backend: str = "transformer_engine",
        cross_attention_backend: str = "transformer_engine",
        natten_params: Mapping | None = None,
        state_t: int = None,  # noqa: RUF013
        enable_cross_view_attn: bool = True,
    ):
        super().__init__(
            x_dim,
            context_dim,
            num_heads,
            mlp_ratio,
            use_adaln_lora,
            adaln_lora_dim,
            self_attention_backend,
            cross_attention_backend,
            natten_params,
        )
        self.state_t = state_t
        self.enable_cross_view_attn = enable_cross_view_attn
        del self.cross_attn
        self.cross_attn = MultiViewCrossAttention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            state_t=state_t,
            backend=cross_attention_backend,
        )

        # hard-coding for cross-view attn here
        cross_view_attn_map = {
            0: [1, 2, 3],
            1: [0, 2, 3],
            2: [0, 1, 3],
            3: [0, 1, 2],
        }
        if enable_cross_view_attn:
            self.cross_view_attn = CrossViewAttention(
                x_dim,
                x_dim,  # context_dim, can not set to None
                num_heads,
                x_dim // num_heads,
                qkv_format="bshd",
                cross_view_attn_map=cross_view_attn_map,
            )
            # no modulation so we set elementwise_affine=True
            self.layer_norm_cross_view_attn = nn.LayerNorm(x_dim, elementwise_affine=True, eps=1e-6)

    def reset_parameters(self):
        super().reset_parameters()
        if self.enable_cross_view_attn:
            self.layer_norm_cross_view_attn.reset_parameters()

    def init_weights(self):
        super().init_weights()
        if self.enable_cross_view_attn:
            self.cross_view_attn.init_weights()

            # Zero-initialize the output projection
            torch.nn.init.zeros_(self.cross_view_attn.output_proj.weight)
            if self.cross_view_attn.output_proj.bias is not None:
                torch.nn.init.zeros_(self.cross_view_attn.output_proj.bias)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        view_indices_B_T: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
        k_mem: torch.Tensor | None = None, # (B, t, D)
        v_mem: torch.Tensor | None = None, # (B, t, D)
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = (
                self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = (
                self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = (
                self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D
            ).chunk(3, dim=-1)
        else:
            shift_self_attn_B_T_D, scale_self_attn_B_T_D, gate_self_attn_B_T_D = self.adaln_modulation_self_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_cross_attn_B_T_D, scale_cross_attn_B_T_D, gate_cross_attn_B_T_D = self.adaln_modulation_cross_attn(
                emb_B_T_D
            ).chunk(3, dim=-1)
            shift_mlp_B_T_D, scale_mlp_B_T_D, gate_mlp_B_T_D = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # Reshape tensors from (B, T, D) to (B, T, 1, 1, D) for broadcasting
        shift_self_attn_B_T_1_1_D = rearrange(shift_self_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_self_attn_B_T_1_1_D = rearrange(scale_self_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_self_attn_B_T_1_1_D = rearrange(gate_self_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_cross_attn_B_T_1_1_D = rearrange(shift_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        scale_cross_attn_B_T_1_1_D = rearrange(scale_cross_attn_B_T_D, "b t d -> b t 1 1 d")
        gate_cross_attn_B_T_1_1_D = rearrange(gate_cross_attn_B_T_D, "b t d -> b t 1 1 d")

        shift_mlp_B_T_1_1_D = rearrange(shift_mlp_B_T_D, "b t d -> b t 1 1 d")
        scale_mlp_B_T_1_1_D = rearrange(scale_mlp_B_T_D, "b t d -> b t 1 1 d")
        gate_mlp_B_T_1_1_D = rearrange(gate_mlp_B_T_D, "b t d -> b t 1 1 d")

        B, T, H, W, D = x_B_T_H_W_D.shape

        def _fn(_x_B_T_H_W_D, _norm_layer, _scale_B_T_1_1_D, _shift_B_T_1_1_D):
            return _norm_layer(_x_B_T_H_W_D) * (1 + _scale_B_T_1_1_D) + _shift_B_T_1_1_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_self_attn,
            scale_self_attn_B_T_1_1_D,
            shift_self_attn_B_T_1_1_D,
        )

        video_size = VideoSize(T=T, H=H, W=W)

        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)


        result_B_T_H_W_D = rearrange( # (B, tl, 15, 20, 2048)
            self.self_attn(
                # normalized_x_B_T_HW_D,
                rearrange(normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D, # (tl*15*20, 1, 1, 128)
                video_size=video_size, # (tl, 15, 20)
                k_mem=k_mem,
                v_mem=v_mem,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_self_attn_B_T_1_1_D * result_B_T_H_W_D

        # I think we should insert cross view attn here. x_B_T_H_W_D
        if self.enable_cross_view_attn:
            num_cameras = torch.unique(view_indices_B_T[0]).shape[0]
            x_B_V_T_H_W_D = rearrange(x_B_T_H_W_D, "b (v t) h w d -> b v t h w d", v=num_cameras)
            sv_video_size = VideoSize(T=x_B_V_T_H_W_D.shape[2], H=x_B_V_T_H_W_D.shape[3], W=x_B_V_T_H_W_D.shape[4])
            x_B_V_L_D = rearrange(x_B_V_T_H_W_D, "b v t h w d -> b v (t h w) d")
            view_indices_B_V = rearrange(view_indices_B_T, "b (v t) -> b v t", v=num_cameras)[..., 0]
            result_cross_view_attn_B_T_H_W_D = rearrange(
                self.cross_view_attn(self.layer_norm_cross_view_attn(x_B_V_L_D), view_indices_B_V, sv_video_size),
                "b v (t h w) d -> b (v t) h w d",
                v=num_cameras,
                t=sv_video_size.T,
                h=sv_video_size.H,
                w=sv_video_size.W,
            )
            x_B_T_H_W_D = x_B_T_H_W_D + result_cross_view_attn_B_T_H_W_D

        def _x_fn(
            _x_B_T_H_W_D: torch.Tensor,
            layer_norm_cross_attn: Callable,
            _scale_cross_attn_B_T_1_1_D: torch.Tensor,
            _shift_cross_attn_B_T_1_1_D: torch.Tensor,
        ) -> torch.Tensor:
            _normalized_x_B_T_H_W_D = _fn(
                _x_B_T_H_W_D, layer_norm_cross_attn, _scale_cross_attn_B_T_1_1_D, _shift_cross_attn_B_T_1_1_D
            )
            _result_B_T_H_W_D = rearrange(
                self.cross_attn(
                    rearrange(_normalized_x_B_T_H_W_D, "b t h w d -> b (t h w) d"),
                    crossattn_emb,
                    rope_emb=rope_emb_L_1_1_D,
                ),
                "b (t h w) d -> b t h w d",
                t=T,
                h=H,
                w=W,
            )
            return _result_B_T_H_W_D

        result_B_T_H_W_D = _x_fn(
            x_B_T_H_W_D,
            self.layer_norm_cross_attn,
            scale_cross_attn_B_T_1_1_D,
            shift_cross_attn_B_T_1_1_D,
        )
        x_B_T_H_W_D = result_B_T_H_W_D * gate_cross_attn_B_T_1_1_D + x_B_T_H_W_D

        normalized_x_B_T_H_W_D = _fn(
            x_B_T_H_W_D,
            self.layer_norm_mlp,
            scale_mlp_B_T_1_1_D,
            shift_mlp_B_T_1_1_D,
        )
        result_B_T_H_W_D = self.mlp(normalized_x_B_T_H_W_D)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp_B_T_1_1_D * result_B_T_H_W_D
        return x_B_T_H_W_D


class ActionConditionedMultiViewDiT(MultiViewDiT):
    def __init__(
        self,
        *args,
        crossattn_emb_channels: int = 1024,
        mlp_ratio: float = 4.0,
        enable_cross_view_attn: bool = True,
        state_t: int,
        n_cameras_emb: int,
        view_condition_dim: int,
        concat_view_embedding: bool,
        layer_mask: list[bool] | None = None,
        sac_config: SACConfig = SACConfig(),  # noqa: B008
        natten_parameters: list[Mapping | None] | None = None,
        **kwargs
    ):
        assert "action_dim" in kwargs, "action_dim must be provided"
        action_dim = kwargs.pop("action_dim")
        use_black = bool(kwargs.pop("use_black", False))
        use_history = bool(kwargs.pop("use_history", False))
        super().__init__(
            *args,
            state_t=state_t,
            n_cameras_emb=n_cameras_emb,
            view_condition_dim=view_condition_dim,
            concat_view_embedding=concat_view_embedding,
            layer_mask=layer_mask,
            sac_config=sac_config,
            natten_parameters=natten_parameters,
            **kwargs,
        )
        self.action_embedder_B_D_agi = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_embedder_B_3D_agi = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels * 3,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.action_to_crossattn_emb = Mlp(
            in_features=action_dim,
            hidden_features=self.model_channels * 4,
            out_features=self.model_channels // 2,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

        self.use_black = use_black
        if self.use_black:
            self.black_to_crossattn_emb = LatentToCrossAttEmb(in_channels=16, mid_channels=512, out_dim=1024, pool_t=16, pool_h=12, pool_w=8)

        self.use_history = use_history
        if self.use_history:
            self.framepack_mem = FramePackMemory(
                in_ch=16,
                D=self.model_channels,
                kernels=((1, 2, 2), (1, 4, 4), (1, 8, 8)),
                lam=2.0,
                max_tokens=1024,
                tail="one",
            )
            self.mem_to_k = nn.Linear(self.model_channels, self.model_channels, bias=False)
            self.mem_to_v = nn.Linear(self.model_channels, self.model_channels, bias=False)

        # dit blocks with multiview-cross attn and cross-view attn
        del self.blocks
        self.blocks = nn.ModuleList(
            [
                MultiViewCrossBlock(
                    x_dim=self.model_channels,
                    context_dim=crossattn_emb_channels,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                    self_attention_backend=self.atten_backend
                    if natten_parameters is None or natten_parameters[i] is None
                    else "natten",
                    cross_attention_backend=self.atten_backend,
                    natten_params=None if natten_parameters is None else natten_parameters[i],
                    state_t=self.state_t,
                    enable_cross_view_attn=enable_cross_view_attn,
                )
                for i in range(self.num_blocks)
            ]
        )

        # initilization
        self.init_weights()
        self.enable_selective_checkpoint(sac_config)

    def build_patch_embed(self) -> None:
        (
            concat_padding_mask,
            in_channels,
            patch_spatial,
            patch_temporal,
            model_channels,
        ) = (
            self.concat_padding_mask,
            self.in_channels,
            self.patch_spatial,
            self.patch_temporal,
            self.model_channels,
        )
        in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.x_embedder_multiview = PatchEmbed(
        # self.x_embedder = PatchEmbed(
            spatial_patch_size=patch_spatial,
            temporal_patch_size=patch_temporal,
            in_channels=in_channels,
            out_channels=model_channels,
        )

    def init_weights(self):
        self.x_embedder_multiview.init_weights()
        # self.x_embedder.init_weights()
        for pos_embedder in self.pos_embedder_options.values():
            pos_embedder.reset_parameters()
        if self.extra_per_block_abs_pos_emb:
            for extra_pos_embedder in self.extra_pos_embedders_options.values():
                extra_pos_embedder.init_weights()

        self.t_embedder[1].init_weights()
        for block in self.blocks:
            block.init_weights()

        self.final_layer.init_weights()
        self.t_embedding_norm.reset_parameters()

    def prepare_embedded_sequence(
        self,
        x_B_C_T_H_W: torch.Tensor,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        view_indices_B_T: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        if self.concat_padding_mask:
            padding_mask = transforms.functional.resize(
                padding_mask, list(x_B_C_T_H_W.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, padding_mask.unsqueeze(1).repeat(1, 1, x_B_C_T_H_W.shape[2], 1, 1)], dim=1
            )
        try:
            process_group = parallel_state.get_context_parallel_group()
            cp_size = len(get_process_group_ranks(process_group))
        except:  # noqa: E722
            cp_size = 1
        n_cameras = (x_B_C_T_H_W.shape[2] * cp_size) // self.state_t
        pos_embedder = self.pos_embedder_options[f"n_cameras_{n_cameras}"]
        if self.concat_view_embedding:
            if view_indices_B_T is None:
                view_indices = torch.arange(n_cameras).clamp(
                    max=self.n_cameras_emb - 1
                )  # View indices [0, 1, ..., V-1]
                view_indices = view_indices.to(x_B_C_T_H_W.device)
                view_embedding = self.view_embeddings(view_indices)  # Shape: [V, embedding_dim]
                view_embedding = rearrange(view_embedding, "V D -> D V")
                view_embedding = (
                    view_embedding.unsqueeze(0).unsqueeze(3).unsqueeze(4).unsqueeze(5)
                )  # Shape: [1, D, V, 1, 1, 1]
            else:
                view_indices_B_T = view_indices_B_T.clamp(max=self.n_cameras_emb - 1)
                view_indices_B_T = view_indices_B_T.to(x_B_C_T_H_W.device).long()
                view_embedding = self.view_embeddings(view_indices_B_T)  # B, (V T), D
                view_embedding = rearrange(view_embedding, "B (V T) D -> B D V T", V=n_cameras)
                view_embedding = view_embedding.unsqueeze(-1).unsqueeze(-1)  # Shape: [B, D, V, T, 1, 1]
            x_B_C_V_T_H_W = rearrange(x_B_C_T_H_W, "B C (V T) H W -> B C V T H W", V=n_cameras)
            view_embedding = view_embedding.expand(
                x_B_C_V_T_H_W.shape[0],
                view_embedding.shape[1],
                view_embedding.shape[2],
                x_B_C_V_T_H_W.shape[3],
                x_B_C_V_T_H_W.shape[4],
                x_B_C_V_T_H_W.shape[5],
            )
            x_B_C_V_T_H_W = torch.cat([x_B_C_V_T_H_W, view_embedding], dim=1)
            x_B_C_T_H_W = rearrange(x_B_C_V_T_H_W, " B C V T H W -> B C (V T) H W", V=n_cameras)

        x_B_T_H_W_D = self.x_embedder_multiview(x_B_C_T_H_W)
        # x_B_T_H_W_D = self.x_embedder(x_B_C_T_H_W)

        if self.extra_per_block_abs_pos_emb:
            extra_pos_embedder = self.extra_pos_embedders_options[str(n_cameras)]
            extra_pos_emb = extra_pos_embedder(x_B_T_H_W_D, fps=fps)
        else:
            extra_pos_emb = None

        if "rope" in self.pos_emb_cls.lower():
            return x_B_T_H_W_D, pos_embedder(x_B_T_H_W_D, fps=fps), extra_pos_emb

        if "fps_aware" in self.pos_emb_cls:
            raise NotImplementedError("FPS-aware positional embedding is not supported for multi-view DIT")

        x_B_T_H_W_D = x_B_T_H_W_D + pos_embedder(x_B_T_H_W_D)

        return x_B_T_H_W_D, None, extra_pos_emb

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,  # (B, 16, tl, hl, wl)；tl = view_num * tl (single)
        timesteps_B_T: torch.Tensor, # (B, tl)
        crossattn_emb: torch.Tensor, # (B, 1536, 1024)
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None, # (B, 1, tl, hl, wl)
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        action: torch.Tensor | None = None,  # (B, T_total, action_dim=14)
        blacks: torch.Tensor | None = None,  # (B, 16, tl, hl, wl)
        history: torch.Tensor | None = None,  # (B, 16, th, hl, wh)
        view_indices_B_T: torch.Tensor | None = None,  # (B, T_total)
        **kwargs,
    ):
        del kwargs
        # 0. concat input masks
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)], dim=1)
        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"

        # 1. project action to action embedding
        assert action is not None, "action must be provided"
        action = rearrange(action, "b t d -> b 1 (t d)")
        action_emb_B_D = self.action_embedder_B_D_agi(action) # (B, 1, 2048)
        action_emb_B_3D = self.action_embedder_B_3D_agi(action) # (B, 1, 2048 * 3)
        if blacks is None or not self.use_black:
            crossattn_emb = self.action_to_crossattn_emb(action) + crossattn_emb
        else:
            crossattn_emb = self.black_to_crossattn_emb(blacks.to(torch.bfloat16)) + crossattn_emb

        # 2. history-aware mechanism
        if self.use_history and history is not None:
            mem_tokens = self.framepack_mem(history.to(torch.bfloat16)) #(B, th, D)
            if mem_tokens is not None:
                k_mem = self.mem_to_k(mem_tokens) #(B, th, D)
                v_mem = self.mem_to_v(mem_tokens) #(B, th, D)
        else:
            k_mem, v_mem = None, None

        # 3. prepare embedding sequence
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
            view_indices_B_T=view_indices_B_T,
        )

        # 4. timestamp embedding
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        # 5. black condition, not used in multi-view mode
        T_target = t_embedding_B_T_D.shape[1]
        cond_D, cond_3D = None, None
        # if self.use_black and blacks is not None:
        #     cond_D, cond_3D = self.action_latent_cond(
        #         blacks=blacks.to(torch.bfloat16),
        #         T_target=T_target,
        #         condition_video_input_mask_B_C_T_H_W=condition_video_input_mask_B_C_T_H_W,
        #     )

        # 6. add action-condition to timestamp embedding
        t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_D  # (B,T,D) + (B,1,D=2048)
        if adaln_lora_B_T_3D is not None:
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + action_emb_B_3D # (B, 1, 6144)
        else:
            adaln_lora_B_T_3D = action_emb_B_3D

        if cond_D is not None and cond_3D is not None:
            t_embedding_B_T_D = t_embedding_B_T_D + cond_D
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + cond_3D
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # 7. for logging purpose
        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb
        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb.shape}"
            )

        # 8. pass latent through timestamp embedding
        if use_cuda_graphs:
            with temporarily_remove_hooks(self):
                shapes_key = create_cuda_graph(
                    self.cuda_graphs,
                    self.blocks,
                    x_B_T_H_W_D,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    rope_emb_L_1_1_D,
                    adaln_lora_B_T_3D,
                    extra_pos_emb,
                    k_mem=k_mem,
                    v_mem=v_mem,
                )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb,
            "k_mem": k_mem,
            "v_mem": v_mem,
        }

        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                view_indices_B_T,
                t_embedding_B_T_D,
                crossattn_emb,
                **block_kwargs,
            )

        # 9. unpachify to original latent size
        x_B_T_H_W_O = self.final_layer(
            x_B_T_H_W_D, #(B, tl, hl//2, wl//2, 2048)
            t_embedding_B_T_D, #(B, tl, 2048)
            adaln_lora_B_T_3D=adaln_lora_B_T_3D, #(B, tl, 2048 * 3)
        )
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O) #(B, 16, tl, hl, wl) <- (B, tl, hl//2, wl//2, 16 * 4)
        return x_B_C_Tt_Hp_Wp
