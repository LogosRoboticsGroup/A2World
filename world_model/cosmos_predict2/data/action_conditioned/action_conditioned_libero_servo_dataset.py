import json
import os
import random
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence, List, Tuple, Literal

import os
import math
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import pickle
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from cosmos_predict2.data.action_conditioned.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    euler2rotm,
    rotm2euler,
)
from cosmos_predict2.data.action_conditioned.self_forcing import SelfForcingFrameLayout

def quats_to_euler(quats, order="xyz", degrees=False):
    q_flat = quats.reshape(-1, 4)
    r = R.from_quat(q_flat)
    euler_flat = r.as_euler(order, degrees=degrees)
    return euler_flat.reshape(quats.shape[0], quats.shape[1], 3)

def save_video_tensor(
    video_thwc: torch.Tensor,
    path: str,
    fps: int = 10,
    codec: str = "libx264",
    bitrate: str | None = None,
) -> str:
    assert video_thwc.ndim == 4 and video_thwc.shape[-1] == 3, \
        f"expect (T,H,W,3), got {tuple(video_thwc.shape)}"

    x = video_thwc.detach()
    if x.is_floating_point():
        x = x.clamp(0, 255).round()
        x = x.to(torch.uint8)
    if x.is_cuda:
        x = x.cpu()
    frames = x.contiguous().numpy()  # (T,H,W,3), uint8, RGB

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    writer_kwargs = dict(fps=fps, codec=codec, macro_block_size=None)
    if bitrate is not None:
        writer_kwargs["bitrate"] = bitrate

    with imageio.get_writer(path, **writer_kwargs) as w:
        for f in frames:
            w.append_data(f)  # f: (H,W,3) uint8 RGB

    return path

def quats_to_euler(quats, order="xyz", degrees=False):
    q_flat = quats.reshape(-1, 4)
    r = R.from_quat(q_flat)
    euler_flat = r.as_euler(order, degrees=degrees)
    return euler_flat.reshape(quats.shape[0], quats.shape[1], 3)

class ResizePadCrop:
    def __init__(self, target_h=240, target_w=320, interp="bilinear"):
        self.target_h = int(target_h)
        self.target_w = int(target_w)
        self.interp = interp

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        assert vid.ndim == 4, f"expect (T,C,H,W), got {tuple(vid.shape)}"
        T, C, H, W = vid.shape
        if H == 0 or W == 0:
            raise ValueError(f"invalid video size: H={H}, W={W}")

        orig_dtype = vid.dtype
        x = vid.to(dtype=torch.float32)

        scale = self.target_h / float(H)
        new_w = max(1, int(round(W * scale)))
        x = F.interpolate(
            x, size=(self.target_h, new_w),
            mode=self.interp, align_corners=False if self.interp in ("bilinear", "bicubic") else None
        )
        if new_w < self.target_w:
            pad_total = self.target_w - new_w
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right, 0, 0), mode="constant", value=0.0)
        elif new_w > self.target_w:
            start = (new_w - self.target_w) // 2
            end = start + self.target_w
            x = x[:, :, :, start:end]

        assert x.shape[2] == self.target_h and x.shape[3] == self.target_w, x.shape
        if orig_dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            x = x.clamp(0, 255).to(orig_dtype)
        return x

def post_process_action(L: Sequence[int], A: np.ndarray, inplace: bool = True) -> np.ndarray:
    L = np.asarray(L)
    if L.ndim != 1:
        raise ValueError(f"L must be 1D, got shape {L.shape}")

    if not isinstance(A, np.ndarray):
        A = np.asarray(A)

    if A.ndim < 1:
        raise ValueError(f"A must have at least 1 dimension, got {A.ndim}")
    if A.shape[0] != len(L):
        raise ValueError(f"Expected A.shape[0]==len(L). Got A.shape[0]={A.shape[0]}, len(L)={len(L)}")

    out = A if inplace else A.copy()

    if len(L) < 2:
        return out

    mask = (L[:-1] == 0) & (L[1:] == 0)  # (T-1,)
    if not np.any(mask):
        return out

    keep_idx = (6, 13)
    if out.ndim < 2:
        raise ValueError(f"A must be at least 2D (T, D) to keep specific dimensions, got shape {out.shape}")
    if max(keep_idx) >= out.shape[1]:
        raise ValueError(f"A.shape[1]={out.shape[1]} is too small for keep_idx={keep_idx}")

    kept = out[:-1][mask][:, keep_idx].copy()
    out[:-1][mask] = 0
    out[:-1][mask][:, keep_idx] = kept
    return out

Strategy = Literal["effort_uniform", "time_uniform", "log_time"]

def servo_based_history_sampling_with_path_collection(
    history_frame_ids: List[int],           # length Th (nodes), may include repeated 0
    history_arm_states: np.ndarray,         # (Th, 6) EDGE servo deltas; last edge is hist_last -> frame_ids[0]
    history_gripper_states: np.ndarray,     # (Th,) EDGE gripper actions aligned with history_arm_states
    frame_ids: List[int],                   # length Tg (nodes)
    arm_states: np.ndarray,                 # (Tg-1, >=6) EDGE servo deltas: frame_ids[i] -> frame_ids[i+1]
    gripper_states: np.ndarray,             # (Tg-1,) EDGE gripper actions aligned with arm_states
    sampled_history_length: int,            # Ts (#history nodes to sample)
    *,
    strategy: Strategy = "effort_uniform",
    w_trans: float = 1.0,
    w_rot: float = 0.3,
    rot_scale: float = 1.0,
    eps: float = 1e-12,
    mean6_tanh_scale: Tuple[float, float] = (0.05, 0.25),  # (trans_scale, rot_scale) for tanh-normalizing mean_u
) -> Tuple[List[int], np.ndarray]:
    """
    Sampling:
      - Choose Ts history NODE indices from [idx0 .. Th-1], must include idx0 (earliest frame_id==0) and Th-1.
      - If short, pad idx0 in the FRONT.

    Action semantics:
      - history_arm_states and arm_states are EDGE servo deltas (NOT additive as pose).
      - Therefore action-path features are LENGTH-INSENSITIVE:
          mean / mean-abs / RMS / bounded ratios + edge endpoints (gripper).

    AP shape:
      - AP: (Tg, Ts, 22)
        22 = mean_u(6) + mean_abs_u(6) + rms_u(6) + len_ratio(1) + g_start,g_end(2) + toggle_ratio(1)

    Edge alignment:
      - For unified nodes: [history nodes 0..Th-1] + [forcing nodes Th..Th+Tg-1]
      - Unified edges index e corresponds to node e -> node e+1
        edges[0..Th-2] : history i -> history i+1  (from history_arm_states[i])
        edges[Th-1]    : history last -> forcing[0] (from history_arm_states[-1])
        edges[Th..]    : forcing i -> forcing i+1  (from arm_states[i])
    """
    # ---------------- checks ----------------
    Ts = int(sampled_history_length)
    if Ts < 1:
        raise ValueError("sampled_history_length (Ts) must be >= 1")
    if len(history_frame_ids) < 1:
        raise ValueError("history_frame_ids must be non-empty")
    if len(frame_ids) < 1:
        raise ValueError("frame_ids must be non-empty")

    H_ids = list(history_frame_ids)
    Th = len(H_ids)
    Tg = len(frame_ids)

    H_u = np.asarray(history_arm_states, dtype=np.float64)
    if H_u.shape != (Th, 6):
        raise ValueError(f"history_arm_states must have shape ({Th},6), got {H_u.shape}")

    H_g = np.asarray(history_gripper_states, dtype=np.float64).reshape(-1)
    if H_g.shape[0] != Th:
        raise ValueError(f"history_gripper_states must have length {Th}, got {H_g.shape[0]}")

    F_u_full = np.asarray(arm_states, dtype=np.float64)
    if F_u_full.ndim != 2 or F_u_full.shape[0] != max(Tg - 1, 0) or F_u_full.shape[1] < 6:
        raise ValueError(f"arm_states must be (Tg-1, >=6). Got {F_u_full.shape}, Tg={Tg}")
    F_u6 = F_u_full[:, :6]  # (Tg-1, 6)

    F_g = np.asarray(gripper_states, dtype=np.float64).reshape(-1)
    if F_g.shape[0] != max(Tg - 1, 0):
        raise ValueError(f"gripper_states must have length Tg-1={max(Tg-1,0)}, got {F_g.shape[0]}")

    # ---------------- sampling (history nodes) ----------------
    try:
        idx0 = H_ids.index(0)
    except ValueError:
        idx0 = 0
    idx_last = Th - 1
    lo, hi = idx0, idx_last
    need_mid = Ts - 2

    def step_effort(u6: np.ndarray) -> float:
        dt = u6[:3]
        dr = u6[3:] * rot_scale
        return float(math.sqrt(w_trans * float(dt @ dt) + w_rot * float(dr @ dr)))

    if Ts == 1:
        chosen = [idx0]
    elif hi <= lo or need_mid <= 0:
        chosen = [idx0] + [idx0] * max(0, need_mid) + [idx_last]
        chosen = chosen[:Ts - 1] + [idx_last]
    else:
        candidates = list(range(lo + 1, hi))
        mids: List[int] = []

        if need_mid > 0 and len(candidates) > 0:
            if strategy == "effort_uniform":
                # arc-length over HISTORY NODES based on HISTORY EDGES
                # edge i corresponds to node i -> i+1 for i=0..Th-2
                seg = np.zeros(Th, dtype=np.float64)  # seg[i] = effort of edge i-1
                for i in range(1, Th):
                    seg[i] = step_effort(H_u[i - 1])
                arc = np.cumsum(seg)  # arc[node]

                arc0, arch = float(arc[lo]), float(arc[hi])
                if abs(arch - arc0) < eps:
                    targets = np.linspace(lo, hi, num=need_mid + 2)[1:-1]
                    mids = [int(round(x)) for x in targets]
                else:
                    targets = np.linspace(arc0, arch, num=need_mid + 2)[1:-1]
                    arc_slice = arc[lo:hi + 1]
                    mids = []
                    for s in targets:
                        j_local = int(np.argmin(np.abs(arc_slice - s)))
                        mids.append(lo + j_local)

                mids = sorted({i for i in mids if lo < i < hi})
                if len(mids) < need_mid:
                    remaining = [i for i in candidates if i not in mids]
                    remaining.sort()
                    mids += remaining[: (need_mid - len(mids))]
                mids = mids[:need_mid]

            elif strategy == "time_uniform":
                targets = np.linspace(lo, hi, num=need_mid + 2)[1:-1]
                mids = [int(round(x)) for x in targets]
                mids = sorted({i for i in mids if lo < i < hi})
                if len(mids) < need_mid:
                    remaining = [i for i in candidates if i not in mids]
                    remaining.sort()
                    mids += remaining[: (need_mid - len(mids))]
                mids = mids[:need_mid]

            elif strategy == "log_time":
                span = hi - lo
                picks = set()
                kpow = 0
                while len(picks) < need_mid and (2 ** kpow) < span + 1:
                    j = hi - (2 ** kpow)
                    if lo < j < hi:
                        picks.add(j)
                    kpow += 1
                mids = sorted(picks)
                if len(mids) < need_mid:
                    remaining = [i for i in candidates if i not in mids]
                    remaining.sort()
                    mids += remaining[: (need_mid - len(mids))]
                mids = mids[:need_mid]
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

        chosen = [idx0] + sorted(mids) + [idx_last]
        if len(chosen) < Ts:
            chosen = [idx0] * (Ts - len(chosen)) + chosen
        if len(chosen) > Ts:
            chosen = chosen[-Ts:]
        chosen[-1] = idx_last

    sampled_history_frame_ids = [H_ids[i] for i in chosen]  # length Ts

    # ---------------- unified timeline (nodes + edges) ----------------
    N_nodes = Th + Tg
    # Edges count must be N_nodes - 1 = Th + Tg - 1
    # history provides Th edges (including bridge), forcing provides (Tg-1) edges
    edges_u = np.concatenate([H_u, F_u6], axis=0)  # (Th + (Tg-1), 6) == (Th+Tg-1, 6)
    edges_g = np.concatenate([H_g, F_g], axis=0)   # (Th+Tg-1,)

    if edges_u.shape[0] != N_nodes - 1 or edges_g.shape[0] != N_nodes - 1:
        raise RuntimeError("Edge count mismatch; check Th/Tg and edge definitions.")

    # prefix sums over EDGES (index e corresponds to node e -> e+1)
    prefix_u   = np.zeros((N_nodes, 6), dtype=np.float64)  # sum_u up to node
    prefix_abs = np.zeros((N_nodes, 6), dtype=np.float64)
    prefix_sq  = np.zeros((N_nodes, 6), dtype=np.float64)
    for e in range(N_nodes - 1):
        u = edges_u[e]
        prefix_u[e + 1]   = prefix_u[e]   + u
        prefix_abs[e + 1] = prefix_abs[e] + np.abs(u)
        prefix_sq[e + 1]  = prefix_sq[e]  + u * u

    # gripper "toggle" on EDGE commands: count changes between consecutive edges
    toggle_prefix = np.zeros((N_nodes,), dtype=np.float64)  # aligned to edges boundary (node index)
    for e in range(1, N_nodes - 1):
        toggle_prefix[e + 1] = toggle_prefix[e] + (1.0 if edges_g[e] != edges_g[e - 1] else 0.0)
    # also carry forward for node 1
    toggle_prefix[1] = 0.0
    # fill any untouched (already zero) - monotonic by construction
    for i in range(2, N_nodes):
        if toggle_prefix[i] < toggle_prefix[i - 1]:
            toggle_prefix[i] = toggle_prefix[i - 1]

    toggles_total = max(float(toggle_prefix[-1]), 1.0)

    # ---------------- build AP ----------------
    Fdim = 22
    AP = np.zeros((Tg, Ts, Fdim), dtype=np.float32)

    trans_tanh, rot_tanh = mean6_tanh_scale

    for ti in range(Tg):
        tgt_node = Th + ti  # node index in unified timeline

        for j, h_node in enumerate(chosen):
            if tgt_node < h_node:
                continue

            L = int(tgt_node - h_node)  # number of edges on path
            denom = float(max(L, 1))

            sum_u = prefix_u[tgt_node] - prefix_u[h_node]
            abs_u = prefix_abs[tgt_node] - prefix_abs[h_node]
            sq_u  = prefix_sq[tgt_node] - prefix_sq[h_node]

            # 1) mean_u (length-insensitive)
            mean_u = sum_u / denom
            mean_u_b = mean_u.copy()
            mean_u_b[:3] = np.tanh(mean_u_b[:3] / max(trans_tanh, 1e-9))
            mean_u_b[3:] = np.tanh(mean_u_b[3:] / max(rot_tanh, 1e-9))

            # 2) mean_abs (length-insensitive)
            mean_abs_u = abs_u / denom

            # 3) rms (length-insensitive)
            rms_u = np.sqrt(np.maximum(sq_u / denom, 0.0))

            # 4) bounded length ratio
            len_ratio = float(L) / float(max(Th + Tg, 1))

            # 5) gripper EDGE endpoints (first/last edge on path)
            if L >= 1:
                g_start = float(edges_g[h_node])         # edge h_node: node h_node -> h_node+1
                g_end   = float(edges_g[tgt_node - 1])   # last edge on path
            else:
                g_start = 0.0
                g_end   = 0.0

            # 6) toggle ratio on EDGE commands (bounded)
            toggles_path = float(toggle_prefix[tgt_node] - toggle_prefix[h_node])
            toggle_ratio = toggles_path / toggles_total

            feat = np.concatenate(
                [
                    mean_u_b,                                     # 6
                    mean_abs_u,                                   # 6
                    rms_u,                                        # 6
                    np.array([len_ratio], dtype=np.float64),       # 1
                    np.array([g_start, g_end], dtype=np.float64),  # 2
                    np.array([toggle_ratio], dtype=np.float64),    # 1
                ],
                axis=0,
            ).astype(np.float32)

            AP[ti, j] = feat

    return sampled_history_frame_ids, AP

caption_to_suite = {
    "pick up the black bowl between the plate and the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl from table center and place it on the plate": "libero_spatial",
    "pick up the black bowl on the cookie box and place it on the plate": "libero_spatial",
    "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate": "libero_spatial",
    "pick up the black bowl on the ramekin and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the cookie box and place it on the plate": "libero_spatial",
    "pick up the black bowl on the stove and place it on the plate": "libero_spatial",
    "pick up the black bowl next to the plate and place it on the plate": "libero_spatial",
    "pick up the black bowl on the wooden cabinet and place it on the plate": "libero_spatial",
    "pick up the alphabet soup and place it in the basket": "libero_object",
    "pick up the cream cheese and place it in the basket": "libero_object",
    "pick up the salad dressing and place it in the basket": "libero_object",
    "pick up the bbq sauce and place it in the basket": "libero_object",
    "pick up the ketchup and place it in the basket": "libero_object",
    "pick up the tomato sauce and place it in the basket": "libero_object",
    "pick up the butter and place it in the basket": "libero_object",
    "pick up the milk and place it in the basket": "libero_object",
    "pick up the chocolate pudding and place it in the basket": "libero_object",
    "pick up the orange juice and place it in the basket": "libero_object",
    "open the middle drawer of the cabinet": "libero_goal",
    "put the bowl on the stove": "libero_goal",
    "put the wine bottle on top of the cabinet": "libero_goal",
    "open the top drawer and put the bowl inside": "libero_goal",
    "put the bowl on top of the cabinet": "libero_goal",
    "push the plate to the front of the stove": "libero_goal",
    "put the cream cheese in the bowl": "libero_goal",
    "turn on the stove": "libero_goal",
    "put the bowl on the plate": "libero_goal",
    "put the wine bottle on the rack": "libero_goal",
    "close the top drawer of the cabinet": "libero_90",
    "close the top drawer of the cabinet and put the black bowl on top of it": "libero_90",
    "put the black bowl in the top drawer of the cabinet": "libero_90",
    "put the butter at the back in the top drawer of the cabinet and close it": "libero_90",
    "put the butter at the front in the top drawer of the cabinet and close it": "libero_90",
    "put the chocolate pudding in the top drawer of the cabinet and close it": "libero_90",
    "open the bottom drawer of the cabinet": "libero_90",
    "open the top drawer of the cabinet": "libero_90",
    "open the top drawer of the cabinet and put the bowl in it": "libero_90",
    "put the black bowl on the plate": "libero_90",
    "put the black bowl on top of the cabinet": "libero_90",
    "put the black bowl at the back on the plate": "libero_90",
    "put the black bowl at the front on the plate": "libero_90",
    "put the middle black bowl on the plate": "libero_90",
    "put the middle black bowl on top of the cabinet": "libero_90",
    "stack the black bowl at the front on the black bowl in the middle": "libero_90",
    "stack the middle black bowl on the back black bowl": "libero_90",
    "put the frying pan on the stove": "libero_90",
    "put the moka pot on the stove": "libero_90",
    "turn on the stove and put the frying pan on it": "libero_90",
    "close the bottom drawer of the cabinet": "libero_90",
    "close the bottom drawer of the cabinet and open the top drawer": "libero_90",
    "put the black bowl in the bottom drawer of the cabinet": "libero_90",
    "put the wine bottle in the bottom drawer of the cabinet": "libero_90",
    "put the wine bottle on the wine rack": "libero_90",
    "put the ketchup in the top drawer of the cabinet": "libero_90",
    "close the microwave": "libero_90",
    "put the yellow and white mug to the front of the white mug": "libero_90",
    "open the microwave": "libero_90",
    "put the white bowl on the plate": "libero_90",
    "put the white bowl to the right of the plate": "libero_90",
    "put the right moka pot on the stove": "libero_90",
    "turn off the stove": "libero_90",
    "put the frying pan on the cabinet shelf": "libero_90",
    "put the frying pan on top of the cabinet": "libero_90",
    "put the frying pan under the cabinet shelf": "libero_90",
    "put the white bowl on top of the cabinet": "libero_90",
    "pick up the alphabet soup and put it in the basket": "libero_90",
    "pick up the cream cheese box and put it in the basket": "libero_90",
    "pick up the ketchup and put it in the basket": "libero_90",
    "pick up the tomato sauce and put it in the basket": "libero_90",
    "pick up the butter and put it in the basket": "libero_90",
    "pick up the milk and put it in the basket": "libero_90",
    "pick up the orange juice and put it in the basket": "libero_90",
    "pick up the alphabet soup and put it in the tray": "libero_90",
    "pick up the butter and put it in the tray": "libero_90",
    "pick up the cream cheese and put it in the tray": "libero_90",
    "pick up the ketchup and put it in the tray": "libero_90",
    "pick up the tomato sauce and put it in the tray": "libero_90",
    "pick up the black bowl on the left and put it in the tray": "libero_90",
    "pick up the chocolate pudding and put it in the tray": "libero_90",
    "pick up the salad dressing and put it in the tray": "libero_90",
    "stack the left bowl on the right bowl and place them in the tray": "libero_90",
    "stack the right bowl on the left bowl and place them in the tray": "libero_90",
    "put the red mug on the left plate": "libero_90",
    "put the red mug on the right plate": "libero_90",
    "put the white mug on the left plate": "libero_90",
    "put the yellow and white mug on the right plate": "libero_90",
    "put the chocolate pudding to the left of the plate": "libero_90",
    "put the chocolate pudding to the right of the plate": "libero_90",
    "put the red mug on the plate": "libero_90",
    "put the white mug on the plate": "libero_90",
    "pick up the book and place it in the front compartment of the caddy": "libero_90",
    "pick up the book and place it in the left compartment of the caddy": "libero_90",
    "pick up the book and place it in the right compartment of the caddy": "libero_90",
    "pick up the yellow and white mug and place it to the right of the caddy": "libero_90",
    "pick up the book and place it in the back compartment of the caddy": "libero_10",
    "pick up the red mug and place it to the right of the caddy": "libero_90",
    "pick up the white mug and place it to the right of the caddy": "libero_90",
    "pick up the book in the middle and place it on the cabinet shelf": "libero_90",
    "pick up the book on the left and place it on top of the shelf": "libero_90",
    "pick up the book on the right and place it on the cabinet shelf": "libero_90",
    "pick up the book on the right and place it under the cabinet shelf": "libero_90",
    "put both the alphabet soup and the tomato sauce in the basket": "libero_10",
    "put both the cream cheese box and the butter in the basket": "libero_10",
    "turn on the stove and put the moka pot on it": "libero_10",
    "put the black bowl in the bottom drawer of the cabinet and close it": "libero_10",
    "put the white mug on the left plate and put the yellow and white mug on the right plate": "libero_10",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate": "libero_10",
    "put both the alphabet soup and the cream cheese box in the basket": "libero_10",
    "put both moka pots on the stove": "libero_10",
    "put the yellow and white mug in the microwave and close it": "libero_10"
  }

class ActionConditionedServoLiberoDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        sequence_interval=1,
        num_frames=21,
        cam_ids=[0],
        accumulate_action=False,
        video_size=[256, 256],
        val_start_frame_interval=1,
        debug=False,
        normalize=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        load_history_video=True,
        use_history_sampling=True,
        use_self_forcing=False,
        self_forcing_min_gap=2,
        self_forcing_max_gap=10,
        history_video_length=60,
        mode="train",
        load_hand=False,
        temporal_multi_view=True,
        is_libero_plus=False,
        servo=True,
    ):
        """Dataset class for loading 3D robot action-conditioned data.

        This dataset loads robot trajectories consisting of RGB video frames, robot states (arm positions and gripper states),
        and computes relative actions between consecutive frames.

        Args:
            train_annotation_path (str): Path to training annotation files
            val_annotation_path (str): Path to validation annotation files
            test_annotation_path (str): Path to test annotation files
            video_path (str): Base path to video files
            sequence_interval (int): Interval between sampled frames in a sequence
            num_frames (int): Number of frames to load per sequence
            cam_ids (list): List of camera IDs to sample from
            accumulate_action (bool): Whether to accumulate actions relative to first frame
            video_size (list): Target size [H,W] for video frames
            val_start_frame_interval (int): Frame sampling interval for validation/test
            debug (bool, optional): If True, only loads subset of data. Defaults to False.
            normalize (bool, optional): Whether to normalize video frames. Defaults to False.
            pre_encode (bool, optional): Whether to pre-encode video frames. Defaults to False.
            do_evaluate (bool, optional): Whether in evaluation mode. Defaults to False.
            load_t5_embeddings (bool, optional): Whether to load T5 embeddings. Defaults to False.
            load_action (bool, optional): Whether to load actions. Defaults to True.
            mode (str, optional): Dataset mode - 'train', 'val' or 'test'. Defaults to 'train'.

        The dataset loads robot trajectories and computes:
        - RGB video frames from specified camera views
        - Robot arm states (xyz position + euler angles)
        - Gripper states (binary open/closed)
        - Relative actions between consecutive frames

        Actions are computed as relative transforms between frames:
        - Translation: xyz offset in previous frame's coordinate frame
        - Rotation: euler angles of relative rotation
        - Gripper: binary gripper state

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - action: Action tensor [T-1,7]
            - video_name: Dict with episode/frame metadata
            - latent: Pre-encoded video features if pre_encode=True
        """

        super().__init__()
        if mode == "train":
            self.start_frame_interval = 3
        elif mode == "val":
            self.start_frame_interval = val_start_frame_interval
        elif mode == "test":
            self.start_frame_interval = val_start_frame_interval
        self.data_path = data_path
        self.video_path = os.path.join(self.data_path, 'clips')
        self.ann_path = os.path.join(self.data_path, 'metadata')
        self.sequence_interval = sequence_interval
        self.mode = mode
        self.sequence_length = num_frames
        self.normalize = normalize
        self.load_action = load_action
        self.pre_encode = False
        self.use_self_forcing = use_self_forcing
        self.use_history_sampling = use_history_sampling
        self.self_forcing_min_gap = int(self_forcing_min_gap)
        self.self_forcing_max_gap = int(self_forcing_max_gap)
        if self.self_forcing_min_gap < 1:
            raise ValueError("self_forcing_min_gap must be positive")
        if self.self_forcing_max_gap < self.self_forcing_min_gap:
            raise ValueError("self_forcing_max_gap must be >= self_forcing_min_gap")
        if self.self_forcing_max_gap >= self.sequence_length:
            raise ValueError("self_forcing_max_gap must be smaller than num_frames")
        if self.use_self_forcing:
            self.load_history_video = True
            self.history_video_length = history_video_length
        else:
            self.load_history_video = load_history_video
            self.history_video_length = history_video_length
        self.servo = servo
        self.is_libero_plus = is_libero_plus

        # task info
        task_info = {}
        task_info_path = os.path.join(self.data_path, 'dataset.json')
        with open(task_info_path, "r") as f:
            task_file = json.load(f)
        for info in task_file:
            caption, media_path = info['caption'], info['media_path']
            media_name = media_path.split('/')[-1].replace('.mp4', '')
            if 'eye_in_hand' in media_name or 'wrist' in media_name:
                continue
            task = caption_to_suite.get(caption, caption)
            task_info[media_name] = task
        self.task_info = task_info

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action
        self.load_t5_embeddings = load_t5_embeddings
        if self.load_t5_embeddings:
            dataset_json_path = os.path.join(data_path, 'dataset.json')
            t5_embedding_dir = os.path.join(data_path, 't5_embeddings')
            instruction_dict = {}
            with open(dataset_json_path, "r") as f:
                contents = json.load(f)
            for content in contents:
                instruction = content['caption']
                media_name = content['media_path'].split('/')[-1].replace('.mp4', '')
                instruction = instruction.strip("\n")
                if instruction.startswith('"') and instruction.endswith('"'):
                    instruction = instruction[1:-1]
                t5_xxl_filename = os.path.join(t5_embedding_dir, instruction.replace(' ', '_') + '.pickle')
                instruction_dict[media_name] = t5_xxl_filename
            self.instruction_dict = instruction_dict
        self.load_hand = load_hand
        self.video_size = video_size
        if self.load_hand:
            self.state_t = (self.sequence_length - 1) // 4 + 1
            self.temporal_multi_view = temporal_multi_view

        self.action_dim = 14  # ee xyz (3) + ee euler (3) + gripper(1) * 2 for 2 arms
        if self.servo:
            self.c_act_scaler = [5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 1, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 1]
        else:
            self.c_act_scaler = [1, 1, 1, 1, 1, 1, 1/0.04, 1, 1, 1, 1, 1, 1, 1/0.04]
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.ann_files = self._init_anns(self.ann_path)

        # print(f"{len(self.ann_files)} trajectories in total")
        self.samples = self._init_sequences(self.ann_files)

        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))
        if debug and not do_evaluate:
            self.samples = self.samples[0:10]
        # print(f"{len(self.ann_files)} trajectories in total")
        # print(f"{len(self.samples)} samples in total")
        # with open('./samples_16.pkl','wb') as file:
        #     pickle.dump(self.samples,file)
        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),  # 288 512
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.data_path}"

    def _init_anns(self, data_dir):
        if self.is_libero_plus:
            ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.endswith(".npz") and 'front_episode' in f)]
        else:
            ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.endswith(".npz") and 'agentview' in f)]
        total_num = len(ann_files)
        if self.mode == 'train':
            # return ann_files[:total_num * 15 // 16]
            return ann_files
        elif self.mode == 'test':
            return ann_files[total_num * 14 // 16: total_num * 15 // 16]
        elif self.mode == 'val':
            return ann_files[total_num * 15 // 16: ]
            # val_ann_files = ann_files[total_num * 15 // 16: ]
            # filtered_ann_files = []
            # for ann_file in val_ann_files:
            #     media_name = ann_file.split('/')[-1].replace('.npz', '')
            #     if self.task_info[media_name] == 'libero_10':
            #         filtered_ann_files.append(ann_file)
            # return filtered_ann_files

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(64) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            # for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
            for future in as_completed(future_to_ann_file):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        ann = np.load(ann_file)

        try:
            n_frames = ann['end_position'].shape[0]
        except:
            n_frames = ann['action'].shape[0]
        for frame_i in range(0, n_frames, self.start_frame_interval):
            sample = dict()
            sample["ann_file"] = ann_file
            sample["frame_ids"] = []
            if self.load_history_video:
                sample["history_frame_ids"] = []

            curr_frame_i = frame_i
            while True:
                if curr_frame_i > (n_frames - 1):
                    break
                sample["frame_ids"].append(curr_frame_i)
                if len(sample["frame_ids"]) == self.sequence_length:
                    break
                curr_frame_i += self.sequence_interval

            if len(sample["frame_ids"]) != self.sequence_length:
                continue

            if self.load_history_video:
                if self.use_history_sampling:
                    # 1. History sampling mode - initially get a non padded real history
                    # anchor=3 -> [0, 1, 2]
                    # anchor=8 -> [0, 1, 2, 3, 4, 5, 6, 7]
                    # the history length will be sampled to a fixed length (self.history_video_length) in __getitem__ based on arm pose
                    anchor = sample["frame_ids"][0]
                    history = list(range(0, anchor))
                    sample["history_frame_ids"] = history
                else:
                    # 2. Basic mode - pad frame id 0 if the available history length is less than self.history_video_length
                    # to get fixed length history frames
                    # situation 1: self.history_video_length=5, anchor=10 -> [5, 6, 7, 8, 9]
                    # situation 2: self.history_video_length=5, anchor=2 -> [0, 0, 0, 0, 1]
                    L = int(self.history_video_length)
                    anchor = sample["frame_ids"][0]
                    start = max(0, anchor - L)
                    history = list(range(start, anchor))
                    if len(history) < L:
                        pad_cnt = L - len(history)
                        history = [0] * pad_cnt + history
                    # if self.use_self_forcing:
                    #     history = history + [anchor]
                    sample["history_frame_ids"] = history
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids):
        from decord import VideoReader, cpu  # Importing here due to malloc errors on ARM when importing on top level

        current_frame_ids, history_frame_ids = frame_ids[0], frame_ids[1]
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(current_frame_ids) < len(vr)).all()
        assert (np.array(current_frame_ids) >= 0).all()
        if history_frame_ids is not None:
            assert (np.array(history_frame_ids) < len(vr)).all()
            assert (np.array(history_frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(current_frame_ids).asnumpy()
        if history_frame_ids is not None:
            history_frame_data = vr.get_batch(history_frame_ids).asnumpy()
        else:
            history_frame_data = None
        return frame_data, history_frame_data

    def _get_frames(self, video_path, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            frames, history_frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
            if history_frames is not None and self.load_history_video:
                history_frames = history_frames.astype(np.uint8)
                history_frames = torch.from_numpy(history_frames).permute(0, 3, 1, 2)

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                writer = imageio.get_writer(filename, fps=10)  # fps
                for frame in t_videos:
                    writer.append_data(frame)  # 1 4 13 23 # fp16 24 76 456 688

            if self.normalize:
                frames = self.preprocess(frames)
                if history_frames is not None and self.load_history_video:
                    history_frames = self.preprocess(history_frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
                if history_frames is not None and self.load_history_video:
                    history_frames = self.not_norm_preprocess(history_frames)
                    history_frames = torch.clamp(history_frames * 255.0, 0, 255).to(torch.uint8)
        return frames, history_frames

    def _get_obs(self, video_path, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames, history_frames = self._get_frames(video_path, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, history_frames, temp_cam_id

    def _get_robot_states(self, label, frame_ids):
        extrinsics = np.array(label['extrinsics']) # (N, 4, 4)
        intrinsics = np.array(label['intrinsics']) # (3, 3)
        end_position = np.array(label['end_position']) # (N, 2, 3)
        end_rotation = quats_to_euler(np.array(label['end_orientation'])) # (N, 2, 4) -> (N, 2, 3)
        effector_position = np.array(label['effector_position'])[..., 0] # (N, 2)
        # effector_position = (effector_position > 0.01) * 1.0

        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
        cont_gripper_states = effector_position[frame_ids]
        arm_states = np.concatenate([end_p, end_s], -1)
        return arm_states, cont_gripper_states  #[:, :, 0]

    def _get_all_robot_states(self, label, frame_ids):
        end_position = np.array(label['end_position']) # (N, 2, 3)
        end_rotation = quats_to_euler(np.array(label['end_orientation'])) # (N, 2, 4) -> (N, 2, 3)
        effector_position = np.array(label['effector_position']) # (N, 2)

        end_p = end_position[frame_ids]
        end_s = end_rotation[frame_ids]
        cont_gripper_states = effector_position[frame_ids]
        arm_states = np.concatenate([end_p, end_s], -1)
        return arm_states, cont_gripper_states

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            for i in range(arm_states.shape[1]):
                first_xyz = arm_states[0, i, 0:3]
                first_rpy = arm_states[0, i, 3:6]
                first_rotm = euler2rotm(first_rpy)
                for k in range(1, self.sequence_length):
                    curr_xyz = arm_states[k, i, 0:3]
                    curr_rpy = arm_states[k, i, 3:6]
                    curr_gripper = gripper_states[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                    rel_rotm = first_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper
        else:
            for i in range(arm_states.shape[1]):
                for k in range(1, arm_states.shape[0]):
                    prev_xyz = arm_states[k - 1, i, 0:3]
                    prev_rpy = arm_states[k - 1, i, 3:6]
                    prev_rotm = euler2rotm(prev_rpy)
                    curr_xyz = arm_states[k, i, 0:3]
                    curr_rpy = arm_states[k, i, 3:6]
                    curr_gripper = gripper_states[k, i]
                    curr_rotm = euler2rotm(curr_rpy)
                    rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                    rel_rotm = prev_rotm.T @ curr_rotm
                    rel_rpy = rotm2euler(rel_rotm)
                    action[k - 1, i*7: i*7 + 3] = rel_xyz
                    action[k - 1, i*7 + 3: i*7 + 6] = rel_rpy
                    action[k - 1, i*7 + 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def __getitem__(self, index, cam_id=None, return_video=False):
        try:
            if self.mode != "train":
                np.random.seed(index)
                random.seed(index)

            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]
            data = dict()
            self_forcing_layout = None
            if self.load_history_video:
                history_frame_ids = list(sample["history_frame_ids"])
                if self.use_self_forcing:
                    gap = random.randint(self.self_forcing_min_gap, self.self_forcing_max_gap)
                    self_forcing_layout = SelfForcingFrameLayout.build(history_frame_ids, frame_ids, gap)
                    history_frame_ids = list(self_forcing_layout.extended_history_frame_ids)
            else:
                history_frame_ids = None

            label = np.load(ann_file)
            video_path = ann_file.replace('metadata', 'clips').replace('.npz', '.mp4')
            black_path = ann_file.replace('metadata', 'blacks').replace('.npz', '_black.mp4')
            if self.load_hand:
                if self.is_libero_plus:
                    wrist_video_path = video_path.replace('front_episode', 'wrist_episode')
                else:
                    wrist_video_path = video_path.replace('agentview', 'eye_in_hand')
                if not os.path.exists(wrist_video_path):
                    raise NotImplementedError("Wrist view is not available.")

            if self.servo:
                # using servo actions
                # In the raw servo action -1 means open, 1 mean close, so we need to change it
                if self.is_libero_plus:
                    raw_servo_actions = np.array(label['action'])
                else:
                    raw_servo_actions = np.array(label['actions'])
                actions = raw_servo_actions[frame_ids][:-1].reshape(self.sequence_length - 1, self.action_dim)
                actions[:, 6] = (1 - actions[:, 6]) / 2
                arm_states, gripper_states = actions[:, :6], actions[:, 6]
                actions = torch.from_numpy(actions)
                if self.use_self_forcing:
                    if self_forcing_layout is None:
                        raise RuntimeError("Self-forcing layout was not initialized")
                    forcing_frame_ids = list(self_forcing_layout.forcing_frame_ids)
                    forcing_actions = raw_servo_actions[forcing_frame_ids][:-1].reshape(self.sequence_length - 1, self.action_dim)
                    forcing_actions[:, 6] = (1 - forcing_actions[:, 6]) / 2
                    forcing_actions = post_process_action(forcing_frame_ids[:-1], forcing_actions)
                    forcing_arm_states, forcing_gripper_states = forcing_actions[:, :6], forcing_actions[:, 6]
                    forcing_actions = torch.from_numpy(forcing_actions)
                    forcing_actions *= self.c_act_scaler
                    # for history sampling, we should addtionally get the pose of each history as the guidance of sampling
                    # this operation should be calculated twice, one for history of self-forcing, another for history of training
                    if self.use_history_sampling:
                        history_frame_ids_forcing = list(self_forcing_layout.forcing_history_frame_ids)
                        history_forcing_actions = raw_servo_actions[history_frame_ids_forcing].reshape(len(history_frame_ids_forcing), self.action_dim)
                        history_forcing_actions[:, 6] = (1 - history_forcing_actions[:, 6]) / 2
                        history_forcing_actions = post_process_action(history_frame_ids_forcing, history_forcing_actions)
                        history_arm_states_forcing, history_gripper_states_forcing = history_forcing_actions[:, :6], history_forcing_actions[:, 6]
                        # sampled_history_frame_ids_forcing, forcing_delta = pose_based_history_sampling(history_frame_ids_forcing, history_arm_states_forcing[:, 0, :], history_length=self.history_video_length, strategy="arc_uniform")
                        sampled_history_frame_ids_forcing, forcing_action_paths = servo_based_history_sampling_with_path_collection(
                            history_frame_ids_forcing, # List, e.g., [0, 0, 0, 1, 2, 3, 4], length Th
                            history_arm_states_forcing, # array, (Th, 6)
                            history_gripper_states_forcing, # array, (Th, )
                            forcing_frame_ids, # List, e.g., [5, 6, 7, 8, 9], length Tg
                            forcing_arm_states, # array, (Tg - 1, 6)
                            forcing_gripper_states, # array, (Tg - 1, )
                            self.history_video_length
                        ) # forcing_action_paths (Tg, Th, 22)
                        history_frame_ids_training = list(self_forcing_layout.training_history_frame_ids)
                        # history_arm_states_training, history_gripper_states_training = self._get_robot_states(label, history_frame_ids_training)
                        history_training_actions = raw_servo_actions[history_frame_ids_training].reshape(len(history_frame_ids_training), self.action_dim)
                        history_training_actions = post_process_action(history_frame_ids_training, history_training_actions)
                        history_training_actions[:, 6] = (1 - history_training_actions[:, 6]) / 2
                        history_arm_states_training, history_gripper_states_training = history_training_actions[:, :6], history_training_actions[:, 6]
                        # sampled_history_frame_ids_training, training_delta = pose_based_history_sampling(history_frame_ids_training, history_arm_states_training[:, 0, :], history_length=self.history_video_length, strategy="arc_uniform")
                        sampled_history_frame_ids_training, training_action_paths = servo_based_history_sampling_with_path_collection(
                            history_frame_ids_training, # List, e.g., [0, 0, 0, 1, 2, 3, 4], length Th
                            history_arm_states_training, # array, (Th, 6)
                            history_gripper_states_training, # array, (Th, )
                            frame_ids, # List, e.g., [5, 6, 7, 8, 9], length Tg
                            arm_states, # array, (Tg - 1, 6)
                            gripper_states, # array, (Tg - 1, )
                            self.history_video_length
                        ) # forcing_action_paths (Tg, Th, 22)

                        data['forcing_predict_frame_ids'] = torch.from_numpy(np.array(forcing_frame_ids))
                        data['sampled_history_frame_ids_training'] = torch.from_numpy(np.array(sampled_history_frame_ids_training))
                        data['sampled_history_frame_ids_forcing'] = torch.from_numpy(np.array(sampled_history_frame_ids_forcing))
                        data['delta_training'] = torch.from_numpy(training_action_paths)
                        data['delta_forcing'] = torch.from_numpy(forcing_action_paths)

                        history_frame_ids = sampled_history_frame_ids_forcing + sampled_history_frame_ids_training + forcing_frame_ids
            else:
                # using delta actions
                arm_states, gripper_states = self._get_robot_states(label, frame_ids)
                actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
                if self.use_self_forcing:
                    if self_forcing_layout is None:
                        raise RuntimeError("Self-forcing layout was not initialized")
                    history_arm_states, history_gripper_states = self._get_robot_states(
                        label,
                        list(self_forcing_layout.forcing_frame_ids),
                    )
                    history_actions = self._get_actions(history_arm_states, history_gripper_states, self.accumulate_action)
                    history_actions *= self.c_act_scaler
            actions *= self.c_act_scaler

            if self.load_action:
                data["action"] = actions.float()
                if self.use_self_forcing:
                    data['history_action'] = forcing_actions.float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                video, history_video, cam_id = self._get_obs(video_path, [frame_ids, history_frame_ids], cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                if self.load_hand:
                    wrist_video, wrist_history_video, cam_id = self._get_obs(wrist_video_path, [frame_ids, history_frame_ids], cam_id, pre_encode=False)
                    wrist_video = wrist_video.permute(1, 0, 2, 3) # (C, T, H, W)
                    if history_video is not None:
                        history_video = history_video.permute(1, 0, 2, 3)
                    if wrist_history_video is not None:
                        wrist_history_video = wrist_history_video.permute(1, 0, 2, 3)
                    if self.mode == 'train':
                        patterns = [
                            [0, 2],
                            [2, 0],
                        ]
                    else:
                        patterns = [
                            [2, 0]
                        ]
                    pattern = random.choice(patterns)
                    third_views = [
                        (video, history_video),
                    ]
                    video_views = []     # list of (C, T, H, W)
                    history_views = []   # list of (C, T, H, W)
                    forcing_views = []
                    if self.use_history_sampling:
                        video_history_views = []
                    for flag in pattern:
                        if flag == 0:
                            # wrist-view
                            video_views.append(wrist_video)
                            if self.load_history_video and wrist_history_video is not None:
                                if self.use_self_forcing:
                                    if self.use_history_sampling:
                                        forcing_history_wrist_video = wrist_history_video[:, :self.history_video_length, ...]
                                        training_history_wrist_video = wrist_history_video[:, self.history_video_length: self.history_video_length*2, ...]
                                        forcing_wrist_video = wrist_history_video[:, self.history_video_length*2:, ...]

                                        forcing_views.append(forcing_wrist_video)
                                        history_views.append(forcing_history_wrist_video)
                                        video_history_views.append(training_history_wrist_video)
                                        # del wrist_history_video
                                    else:
                                        # this is for the forcing video
                                        # 1. self-forcing situation
                                        # the self-forcing frames that should be inferenced,
                                        # Build the forcing rollout from the configured gap and target prefix.
                                        # the frame_ids for forcing inference [4, 5, 6] + [7, 8], thus we get frames of this frame_ids
                                        gap = self_forcing_layout.gap
                                        forcing_views.append(torch.cat([wrist_history_video[:, -gap:, ...], wrist_video[:, :self.sequence_length - gap,...]], 1))
                                        # this is for the history for forcing video
                                        # 1. self-forcing situation
                                        # the history frame_ids of the forcing inference [0, 0, 1, 2, 3]
                                        history_views.append(wrist_history_video[:, :-gap, ...])
                                else:
                                    history_views.append(wrist_history_video)
                        elif flag == 2:
                            cur_video, cur_hist = third_views[0]
                            video_views.append(cur_video)
                            if self.load_history_video and cur_hist is not None:
                                if self.use_self_forcing:
                                    if self.use_history_sampling:
                                        forcing_history_video = history_video[:, :self.history_video_length, ...]
                                        training_history_video = history_video[:, self.history_video_length: self.history_video_length*2, ...]
                                        forcing_video = history_video[:, self.history_video_length*2:, ...]

                                        forcing_views.append(forcing_video)
                                        history_views.append(forcing_history_video)
                                        video_history_views.append(training_history_video)
                                        # del history_video
                                    else:
                                        # this is for the forcing video
                                        gap = self_forcing_layout.gap
                                        forcing_views.append(torch.cat([cur_hist[:, -gap:, ...], cur_video[:, :self.sequence_length - gap,...]], 1))
                                        # this is for the history for forcing video
                                        history_views.append(cur_hist[:, :-gap, ...])
                                else:
                                    history_views.append(cur_hist)

                        video_views = [v.to(dtype=torch.uint8) for v in video_views]
                        if not self.temporal_multi_view:
                            video_cat = torch.cat(video_views, dim=-1)  # (C, T, H, 3*W)
                            # data["first_frame"] = video_cat[:, 0, ...]
                        else:
                            video_cat = torch.cat(video_views, dim=1)  # (C, 3*T, H, W)
                            T = video_cat.shape[1] // 2
                            # data["first_frame"] = video_cat[:, [0, T], :, :]
                        data["video"] = video_cat
                        data["view_indices"] = torch.tensor(pattern).repeat_interleave(self.sequence_length).contiguous()
                        data["latent_view_indices_B_T"] = torch.tensor(pattern).repeat_interleave(self.state_t).contiguous()

                        if self.load_history_video and history_video is not None and len(history_views) == 2:
                            history_views = [h.to(dtype=torch.uint8) for h in history_views]
                            if not self.temporal_multi_view:
                                history_cat = torch.cat(history_views, dim=-1)  # (C, T, H, V*W)
                                if self.use_history_sampling:
                                    video_history_cat = torch.cat(video_history_views, dim=-1)
                            else:
                                history_cat = torch.cat(history_views, dim=1) # (C, V*T, H, W)
                                if self.use_history_sampling:
                                    video_history_cat = torch.cat(video_history_views, dim=1)
                            data["history"] = history_cat
                            if self.use_history_sampling:
                                data['video_history'] = video_history_cat

                            if self.use_self_forcing:
                                forcing_views = [h.to(dtype=torch.uint8) for h in forcing_views]
                                if not self.temporal_multi_view:
                                    forcing_cat = torch.cat(forcing_views, dim=-1)  # (C, T, H, V*W)
                                else:
                                    forcing_cat = torch.cat(forcing_views, dim=1) # (C, V*T, H, W)
                                data["forcing_video"] = forcing_cat
                else:
                    data["video"] = video.to(dtype=torch.uint8)
                    data['first_frame'] = video[:, 0, ...]
                    if self.load_history_video and history_video is not None:
                        history_video = history_video.permute(1, 0, 2, 3)
                        data['history'] = history_video.to(torch.uint8)

            data["annotation_file"] = ann_file
            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_id" in label:
                data["__key__"] = label["episode_id"]
            else:
                data["__key__"] = video_path

            # add task info
            media_name = video_path.split('/')[-1].replace('.mp4', '')
            data["__task__"] = self.task_info[media_name]

            # Just add these to fit the interface
            if self.load_t5_embeddings:
                t5_embeddings_path = self.instruction_dict[media_name]
                with open(t5_embeddings_path, 'rb') as f:
                    t5_embeddings = pickle.load(f)
                    t5_embeddings = torch.from_numpy(t5_embeddings[0])
                t5 = torch.zeros(512, 1024, dtype=torch.bfloat16)
                t5[:t5_embeddings.shape[0], :] = t5_embeddings
                t5_mask = torch.ones(512, dtype=torch.int64)
                t5_mask[:t5_embeddings.shape[0]] = 0
                if self.load_hand:
                    data["t5_text_embeddings"] = torch.cat([t5, t5], 0).to(torch.bfloat16)
                    data['t5_text_mask'] = torch.cat([t5_mask, t5_mask], 0).to(torch.int64)
                else:
                    data["t5_text_embeddings"] = t5.to(torch.bfloat16)
                    data['t5_text_mask'] = t5_mask.to(torch.int64)
            else:
                if self.load_hand:
                    data["t5_text_embeddings"] = torch.zeros(512 * 2, 1024, dtype=torch.bfloat16)
                    data["t5_text_mask"] = torch.ones(512 * 2, dtype=torch.int64)
                else:
                    data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
                    data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
            data["fps"] = 10
            data["image_size"] = 256 * torch.ones(4)  # TODO: Does this matter?
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256)
            if self.use_self_forcing:
                data['history_video_length'] = self.history_video_length
                if self_forcing_layout is None:
                    raise RuntimeError("Self-forcing layout was not initialized")
                data['self_forcing_gap'] = self_forcing_layout.gap
                data['target_condition_frame_id'] = self_forcing_layout.target_condition_frame_id
            if self.load_hand:
                data["video_name"] = {
                    "video_path": video_path,
                    "start_frame_id": str(frame_ids[0]),
                }
                data["sample_n_views"] = 2
                data["ref_cam_view_idx_sample_position"] = torch.ones(1, dtype=torch.int64) * (-1)
                data["image_size"] = torch.tensor([self.video_size[0], self.video_size[1], self.video_size[0], self.video_size[1]])
                data["padding_mask"] = torch.zeros(1, self.video_size[0], self.video_size[1])

            return data
        except Exception:
            warnings.warn(  # noqa: B028
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")  # noqa: B028
            warnings.warn(traceback.format_exc())  # noqa: B028
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]
