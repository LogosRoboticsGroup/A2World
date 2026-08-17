# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import attrs
import torch
from megatron.core import parallel_state
from torch.distributed.device_mesh import init_device_mesh
from typing import Any
from torch.distributed.tensor import DTensor
import torch.distributed as dist

from cosmos_predict2.models.video2world_model import Predict2Video2WorldModel, Predict2Video2WorldModelConfig
from cosmos_predict2.models.self_forcing import (
    SelfForcingBatch,
    SelfForcingSchedule,
    build_training_batch,
    inject_generated_condition,
    synchronized_bernoulli,
    synchronized_randint,
)
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline, Video2WorldActionConditionedBatchPipeline
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc

import os
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import cv2
import imageio
from imaginaire.utils.io import save_image_or_video
from imaginaire.utils.easy_io import easy_io
from imaginaire.constants import get_cosmos_predict2_multiview_checkpoint, get_cosmos_predict2_video2world_checkpoint

from cosmos_predict2.conditioner import DataType, TextCondition
from cosmos_predict2.pipelines.multiview import MultiviewPipeline
from cosmos_predict2.pipelines.video2world_multiview_action_state_pred import MultiviewVideo2WorldActionConditionedStatePredPipeline
from cosmos_predict2.configs.base.config_multiview_action import (
    MultiviewStatePredPipelineConfig,
    get_cosmos_predict2_multiview_state_pred_pipeline,
)

import time

from cosmos_predict2.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
    split_inputs_cp,
)

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

# optical flow-guided generation evaluation
@torch.no_grad()
def _to_float01_gpu(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.max() > 1.0:
        x = x / 255.0
    return x.clamp(0, 1)

def _gaussian_window_1d(kernel_size=11, sigma=1.5, device="cuda", dtype=torch.float32):
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1)/2
    g = torch.exp(-(coords**2) / (2*sigma*sigma))
    g = g / g.sum()
    return g

def _create_gaussian_kernel_2d(kernel_size=11, sigma=1.5, channels=3, device="cuda", dtype=torch.float32):
    g1 = _gaussian_window_1d(kernel_size, sigma, device, dtype)
    g2 = g1[:, None] @ g1[None, :]
    g2 = g2.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return g2  # (C,1,kh,kw)

def _ssim_gpu(x, y, kernel, C1=0.01**2, C2=0.03**2, eps=1e-12):
    C = x.shape[1]
    padding = kernel.shape[-1] // 2
    # depthwise conv
    mu_x = F.conv2d(x, kernel, padding=padding, groups=C)
    mu_y = F.conv2d(y, kernel, padding=padding, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=padding, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=padding, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=padding, groups=C) - mu_xy

    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + eps)
    return ssim_map.mean(dim=(1,2,3))

def psnr_from_mse(mse, eps=1e-12):
    return 10.0 * torch.log10(1.0 / torch.clamp(mse, min=eps))

@torch.no_grad()
def eval_videos_cuda(gen: torch.Tensor, gt: torch.Tensor, lpips_net: str | None = None):
    assert gen.device.type == "cuda" and gt.device.type == "cuda"
    assert gen.shape == gt.shape and gen.dim() == 4 and gen.size(-1) == 3

    gen = _to_float01_gpu(gen)
    gt  = _to_float01_gpu(gt)

    T, H, W, _ = gen.shape
    gen_chw = gen.permute(0,3,1,2).contiguous()
    gt_chw  = gt.permute(0,3,1,2).contiguous()

    mse_per = F.mse_loss(gen_chw, gt_chw, reduction="none").mean(dim=(1,2,3))   # (T,)
    psnr_per = psnr_from_mse(mse_per)

    kernel = _create_gaussian_kernel_2d(kernel_size=11, sigma=1.5, channels=3,
                                        device=gen.device, dtype=gen_chw.dtype)
    ssim_per = _ssim_gpu(gen_chw, gt_chw, kernel)  # (T,)

    lpips_avg = None
    if lpips_net is not None:
        try:
            import lpips
            loss_fn = lpips.LPIPS(net=lpips_net).to(gen.device)
            to_lpips = lambda x: x.mul(2).sub(1)
            bs = 16
            vals = []
            for i in range(0, T, bs):
                v = loss_fn(to_lpips(gen_chw[i:i+bs]), to_lpips(gt_chw[i:i+bs]))
                vals.append(v.view(-1))
            lpips_avg = torch.cat(vals, 0).mean().item()
        except Exception:
            lpips_avg = None

    if T > 1:
        gen_dt = gen_chw[1:] - gen_chw[:-1]
        gt_dt  = gt_chw[1:]  - gt_chw[:-1]
        tssim_per = _ssim_gpu(gen_dt, gt_dt, kernel)  # (T-1,)
        tssim_avg = tssim_per.mean().item()
    else:
        tssim_per = torch.empty(0, device=gen.device)
        tssim_avg = None

    out = {
        "MSE_mean":  mse_per.mean().item(),
        "PSNR_mean": psnr_per.mean().item(),
        "SSIM_mean": ssim_per.mean().item(),
        "tSSIM_mean": tssim_avg,
    }
    return out


def compute_flows_opencv(frames_rgb):
    """(T,H,W,3)->list of (H,W,2) float32 using Farnebäck"""
    flows = []
    prev = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for t in range(1, len(frames_rgb)):
        nxt = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, nxt, None,
                                            pyr_scale=0.5, levels=3,
                                            winsize=15, iterations=3,
                                            poly_n=5, poly_sigma=1.2, flags=0)
        flows.append(flow.astype(np.float32))
        prev = nxt
    return flows

def flow_bgr_vis(flow):
    fx, fy = flow[...,0], flow[...,1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[...,0] = (ang / (2*np.pi) * 180).astype(np.uint8)
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def compare_videos_with_flow(video_real, video_gen):
    f1 = compute_flows_opencv(video_real)
    f2 = compute_flows_opencv(video_gen)
    n = min(len(f1), len(f2))
    epe_list, cos_list = [], []
    flow_a_list, flow_b_list, epe_vis_list = [], [], []
    for i in range(n):
        a, b = f1[i], f2[i]
        diff = a - b
        epe = np.sqrt((diff**2).sum(-1))
        epe_list.append(float(epe.mean()))

        na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
        cos = ((a*b).sum(-1)) / (na*nb + 1e-6)
        valid = (na > 1e-3) & (nb > 1e-3)
        cos_list.append(float(np.mean(cos[valid])) if np.any(valid) else np.nan)

        flow_a_list.append(flow_bgr_vis(a))
        flow_b_list.append(flow_bgr_vis(b))
        epe_norm = cv2.normalize(epe, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        epe_vis_list.append(cv2.applyColorMap(epe_norm, cv2.COLORMAP_JET))

    arr = np.asarray(cos_list, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.any():
        with np.errstate(all='ignore'):
            dir_cos_mean   = float(np.nanmean(arr[valid]))
            dir_cos_median = float(np.nanmedian(arr[valid]))
    else:
        dir_cos_mean = dir_cos_median = 0.0

    return {
        "pairs": n,
        "EPE_mean": float(np.mean(epe_list)),
        "EPE_median": float(np.median(epe_list)),
        "dir_cos_mean": dir_cos_mean,
        "dir_cos_median": dir_cos_median,
        "flow_a": np.array(flow_a_list),
        "flow_b": np.array(flow_b_list),
        "epe": np.array(epe_vis_list),
    }


@attrs.define(slots=False)
class Predict2ModelManagerConfig:
    # Local path, use it in fast debug run
    # dit_path: str = get_cosmos_predict2_multiview_checkpoint(model_size="2B") # no pretrained multiview model available
    dit_path : str = get_cosmos_predict2_video2world_checkpoint(model_size="2B")
    # For inference
    text_encoder_path: str = ""  # not used in training.

@attrs.define(slots=False)
class Predict2ActionConditionedMultiviewModelConfig:
    train_architecture: str = "base"
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_target_modules: str = "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
    init_lora_weights: bool = True

    precision: str = "bfloat16"
    input_video_key: str = "video"
    input_image_key: str = "images"
    loss_reduce: str = "mean"
    loss_scale: float = 10.0

    adjust_video_noise: bool = True

    # This is used for the original way to load models
    model_manager_config: Predict2ModelManagerConfig = Predict2ModelManagerConfig()  # noqa: RUF009
    # This is a new way to load models
    pipe_config: MultiviewStatePredPipelineConfig = get_cosmos_predict2_multiview_state_pred_pipeline(  # noqa: RUF009
        model_size="2B", views=3, frames=21, fps=10
    )
    # debug flag
    debug_without_randomness: bool = False
    fsdp_shard_size: int = 0  # 0 means not using fsdp, -1 means set to world size
    # High sigma strategy
    high_sigma_ratio: float = 0.0
    self_forcing_warmup_steps: int = 2000
    self_forcing_probability: float = 0.4
    self_forcing_min_sampling_steps: int = 1
    self_forcing_max_sampling_steps: int = 5

class Predict2Video2WorldActionConditionedMultiviewStatePredModel(Predict2Video2WorldModel):
    def __init__(self, config: Predict2ActionConditionedMultiviewModelConfig):
        super(Predict2Video2WorldModel, self).__init__()

        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        self.device = torch.device("cuda")
        self.self_forcing_schedule = SelfForcingSchedule(
            warmup_steps=config.self_forcing_warmup_steps,
            probability=config.self_forcing_probability,
            min_sampling_steps=config.self_forcing_min_sampling_steps,
            max_sampling_steps=config.self_forcing_max_sampling_steps,
        )

        # 1. set data keys and data information
        self.setup_data_key()

        # 4. Set up loss options, including loss masking, loss reduce and loss scaling
        self.loss_reduce = getattr(config, "loss_reduce", "mean")
        assert self.loss_reduce in ["mean", "sum"]
        self.loss_scale = getattr(config, "loss_scale", 1.0)
        log.critical(f"Using {self.loss_reduce} loss reduce with loss scale {self.loss_scale}")
        if self.config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(self.config.pipe_config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        # 7. training states
        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

        # New way to init pipe
        self.pipe = MultiviewVideo2WorldActionConditionedStatePredPipeline.from_config(
            config.pipe_config,
            dit_path=config.model_manager_config.dit_path,
        )

        self.freeze_parameters()
        if config.train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.dit,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_target_modules=config.lora_target_modules,
                init_lora_weights=config.init_lora_weights,
            )
            if self.pipe.dit_ema:
                self.add_lora_to_model(
                    self.pipe.dit_ema,
                    lora_rank=config.lora_rank,
                    lora_alpha=config.lora_alpha,
                    lora_target_modules=config.lora_target_modules,
                    init_lora_weights=config.init_lora_weights,
                )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        total_params = sum(p.numel() for p in self.parameters())
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Print the number in billions, or in the format of 1,000,000,000
        log.info(
            f"Total parameters: {total_params / 1e9:.2f}B, Frozen parameters: {frozen_params:,}, Trainable parameters: {trainable_params:,}"
        )

        if config.fsdp_shard_size != 0 and torch.distributed.is_initialized():
            if config.fsdp_shard_size == -1:
                fsdp_shard_size = torch.distributed.get_world_size()
                replica_group_size = 1
            else:
                fsdp_shard_size = min(config.fsdp_shard_size, torch.distributed.get_world_size())
                replica_group_size = torch.distributed.get_world_size() // fsdp_shard_size
            dp_mesh = init_device_mesh(
                "cuda", (replica_group_size, fsdp_shard_size), mesh_dim_names=("replicate", "shard")
            )
            log.info(f"Using FSDP with shard size {fsdp_shard_size} | device mesh: {dp_mesh}")
            self.pipe.apply_fsdp(dp_mesh)
        else:
            log.info("FSDP (Fully Sharded Data Parallel) is disabled.")

    def forward(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        return super().forward(data_batch, data_batch_idx)

    def state_dict(self) -> dict[str, Any]:
        net_state_dict = self.pipe.dit.state_dict(prefix="net.")
        if self.config.pipe_config.ema.enabled:
            ema_state_dict = self.pipe.dit_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)

        cleaned_state_dict: Dict[str, Any] = {}

        for key, val in net_state_dict.items():
            if isinstance(val, DTensor):
                cleaned_state_dict[key] = val.full_tensor().detach().cpu()
            elif torch.is_tensor(val):
                cleaned_state_dict[key] = val.detach().cpu()
            else:
                cleaned_state_dict[key] = val

        return cleaned_state_dict

    @torch.no_grad()
    def validation_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        video_tensor = data_batch['video'][0].detach().clone() # (3, T*V, H, W)
        n_views = int(data_batch["sample_n_views"][0].detach().cpu().numpy())
        latent_view_indices_B_T = data_batch["latent_view_indices_B_T"][0]
        input_action = data_batch['action'][0].float().detach().cpu().numpy()
        if 'blacks' in data_batch.keys():
            input_blacks = data_batch['blacks'].float().detach().cpu().numpy()
        else:
            input_blacks = None
        if 'video_history' in data_batch.keys():
            input_history = data_batch['video_history'][0].unsqueeze(0)
        else:
            input_history = None
        if 'delta_training' in data_batch.keys():
            input_state = data_batch['delta_training'].float().detach().cpu().numpy()
            input_state = input_state[0][None, ...]
        else:
            input_state = None
        # (1, 3, T*V, H, W)
        video = self.pipe(video_tensor, None, actions=input_action, state=input_state, blacks=input_blacks, history=input_history, latent_view_indices=latent_view_indices_B_T, num_conditional_frames=1, guidance=0, n_views=n_views, use_cuda_graphs=False)
        output_batch, kendall_loss = self.training_step(data_batch, data_batch_idx)
        output_batch['generated_video'] = video
        return output_batch, kendall_loss

    def on_validation_step_end(
        self,
        model: ImaginaireModel,
        data_batch: dict[str, torch.Tensor],
        output_batch: dict[str, torch.Tensor],
        loss: torch.Tensor,
        iteration: int = 0,
    ) -> None:
        path_local = output_batch['path_local']
        video_dir = os.path.join(path_local, 'videos', str(iteration).zfill(5))
        gt_video_dir = os.path.join(path_local, 'videos_gt', str(iteration).zfill(5))
        os.makedirs(video_dir, exist_ok=True)
        os.makedirs(gt_video_dir, exist_ok=True)

        n_views = int(data_batch["sample_n_views"][0].detach().cpu().numpy())
        video = rearrange(output_batch['generated_video'], 'b c (v t) h w -> b c t h (v w)', v=n_views)
        gt_video = rearrange(data_batch['video'], 'b c (v t) h w -> b c t h (v w)', v=n_views)

        production = False
        if 'use_action_bias' in data_batch.keys():
            production = True
            use_action_bias = bool(data_batch['use_action_bias'][0])
        if '__task__' in data_batch.keys():
            task = data_batch['__task__'][0]
            task_suffix = '_' + task
        else:
            task = 'total'
            task_suffix = ''

        if not production:
            samples_name = data_batch['__key__'][0].split('/')[-1].replace('.mp4', '') + task_suffix + '.mp4'
        elif use_action_bias:
            samples_name = data_batch['__key__'][0].split('/')[-1].replace('.mp4', '') + task_suffix + '_biased.mp4'
        elif not use_action_bias:
            samples_name = data_batch['__key__'][0].split('/')[-1].replace('.mp4', '') + task_suffix + '_origin.mp4'
        output_path = os.path.join(video_dir, samples_name)
        save_image_or_video(video, output_path, fps=4)
        gt_output_path = os.path.join(gt_video_dir, samples_name)
        save_image_or_video(gt_video, gt_output_path, fps=4)

        # evaling video quality
        if not production:
            gt_video = ((gt_video[0] + 1.0) / 2.0).clamp(0, 1)
            gt_video = rearrange((gt_video * 255), "c t h w -> t h w c") + 0.5
            eval_video = ((video[0] + 1.0) / 2.0).clamp(0, 1)
            eval_video = rearrange((eval_video * 255), "c t h w -> t h w c") + 0.5 # (F + 1, S, S, 3)
            eval_result = eval_videos_cuda(eval_video, gt_video)
            eval_result['latent_loss'] = loss

            # eval flow
            flow_result = compare_videos_with_flow(eval_video.float().detach().cpu().numpy().astype(np.uint8), gt_video.float().detach().cpu().numpy().astype(np.uint8))
            flow_dir = os.path.join(path_local, 'flows', str(iteration).zfill(5))
            gt_flow_dir = os.path.join(path_local, 'flows_gt', str(iteration).zfill(5))
            epe_vis_dir = os.path.join(path_local, 'epe_vis', str(iteration).zfill(5))
            os.makedirs(flow_dir, exist_ok=True)
            os.makedirs(gt_flow_dir, exist_ok=True)
            os.makedirs(epe_vis_dir, exist_ok=True)
            # samples_name = data_batch['__key__'][0].split('/')[-1]
            flow_out_path = os.path.join(flow_dir, samples_name)
            gt_flow_out_path = os.path.join(gt_flow_dir, samples_name)
            epe_out_path = os.path.join(epe_vis_dir, samples_name)
            easy_io.dump(flow_result['flow_a'], flow_out_path, file_format="mp4", format="mp4", fps=4)
            easy_io.dump(flow_result['flow_b'], gt_flow_out_path, file_format="mp4", format="mp4", fps=4)
            easy_io.dump(flow_result['epe'], epe_out_path, file_format="mp4", format="mp4", fps=4)

            # merge eval results
            eval_result['EPE_mean'] = flow_result['EPE_mean']
            eval_result['EPE_median'] = flow_result['EPE_median']
            eval_result['dir_cos_mean'] = flow_result['dir_cos_mean']
            eval_result['dir_cos_median'] = flow_result['dir_cos_median']
        else:
            eval_result = {}
            eval_result['latent_loss'] = loss
        return eval_result, samples_name

    def training_step(self, data_batch: dict, data_batch_idx: int) -> tuple[dict, torch.Tensor]:
        self.pipe.device = self.device
        probability = self.self_forcing_schedule.probability_at(data_batch_idx)
        use_self_forcing = synchronized_bernoulli(probability, self.device)
        data_batch = self.prepare_training_condition(
            data_batch,
            use_self_forcing=use_self_forcing,
            iteration=data_batch_idx,
        )

        # Loss
        self._update_train_stats(data_batch)

        # Get the input data to noise and denoise~(image, video) and the corresponding conditioner.
        _, x0_B_C_T_H_W, condition = self.pipe.get_data_and_condition(data_batch)

        # Sample pertubation noise levels and N(0, 1) noises
        sigma_B_T, epsilon_B_C_T_H_W = self.draw_training_sigma_and_epsilon(x0_B_C_T_H_W.size(), condition)

        # Broadcast and split the input data and condition for model parallelism
        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T = self.pipe.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )
        output_batch, kendall_loss, _, _ = self.compute_loss_with_epsilon_and_sigma(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, sigma_B_T
        )

        if self.loss_reduce == "mean":
            kendall_loss = kendall_loss.mean() * self.loss_scale
        elif self.loss_reduce == "sum":
            kendall_loss = kendall_loss.sum(dim=1).mean() * self.loss_scale
        else:
            raise ValueError(f"Invalid loss_reduce: {self.loss_reduce}")
        return output_batch, kendall_loss

    @torch.no_grad()
    def prepare_training_condition(
        self,
        data_batch: dict,
        *,
        use_self_forcing: bool,
        iteration: int,
    ) -> dict:
        if "delta_training" not in data_batch:
            if use_self_forcing:
                raise KeyError("Self-forcing was selected, but the dataset did not provide self-forcing metadata")
            return dict(data_batch)

        batch = SelfForcingBatch.from_data_batch(data_batch)
        if not use_self_forcing:
            return build_training_batch(
                data_batch,
                video=batch.target_video,
                history=batch.training_history,
                state=batch.training_state,
            )

        num_steps = synchronized_randint(
            self.self_forcing_schedule.min_sampling_steps,
            self.self_forcing_schedule.max_sampling_steps,
            self.device,
        )
        generated_video = self._generate_self_forcing_rollout(
            batch,
            num_sampling_steps=num_steps,
            seed=iteration,
        )
        video, history = inject_generated_condition(batch, generated_video)
        log.info(
            f"[SelfForcing] iteration={iteration} sampling_steps={num_steps} "
            f"gaps={batch.gaps.tolist()}"
        )
        return build_training_batch(
            data_batch,
            video=video,
            history=history,
            state=batch.training_state,
        )

    @torch.no_grad()
    def get_generated_condition(self, data_batch: dict, use_self_forcing: bool):
        return self.prepare_training_condition(
            data_batch,
            use_self_forcing=use_self_forcing,
            iteration=0,
        )

    @torch.no_grad()
    def _generate_self_forcing_rollout(
        self,
        batch: SelfForcingBatch,
        *,
        num_sampling_steps: int,
        seed: int,
    ) -> torch.Tensor:
        batch_size, channels, total_frames, height, width = batch.forcing_video.shape
        forcing_video = batch.forcing_video.clone().view(
            batch_size,
            channels,
            batch.n_views,
            batch.target_frames,
            height,
            width,
        )
        forcing_video[:, :, :, 1:] = 0
        forcing_video = forcing_video.flatten(2, 3)

        num_latent_conditional_frames = self.pipe.tokenizer.get_latent_num_frames(1)
        forcing_data_batch = self.pipe._get_data_batch_input(
            forcing_video.to(device=self.device),
            actions=batch.forcing_actions,
            state=batch.forcing_state,
            blacks=None,
            history=batch.forcing_history,
            latent_view_indices=batch.latent_view_indices,
            prompt="",
            negative_prompt="",
            num_latent_conditional_frames=num_latent_conditional_frames,
            n_views=batch.n_views,
            fps=batch.fps,
        )
        self.pipe._normalize_video_databatch_inplace(forcing_data_batch)
        self.pipe._augment_image_dim_inplace(forcing_data_batch)
        if "history" in forcing_data_batch:
            self.pipe._normalize_video_databatch_inplace_black_history(
                forcing_data_batch,
                input_key="history",
                check_key="is_history_preprocessed",
            )
            if not forcing_data_batch.get("is_history_latent", False):
                forcing_data_batch["history"] = self.pipe.encode(forcing_data_batch["history"]).contiguous().float()
                forcing_data_batch["is_history_latent"] = True

        input_key = self.pipe.input_image_key if self.pipe.is_image_batch(forcing_data_batch) else self.pipe.input_video_key
        sample_count = forcing_data_batch[input_key].shape[0]
        _, latent_height, latent_width = forcing_data_batch[input_key].shape[-3:]
        state_shape = (
            self.pipe.config.state_ch,
            self.pipe.config.state_t * batch.n_views,
            latent_height // self.pipe.tokenizer.spatial_compression_factor,
            latent_width // self.pipe.tokenizer.spatial_compression_factor,
        )
        x0_fn = self.pipe.get_x0_fn_from_batch(
            forcing_data_batch,
            guidance=0,
            is_negative_prompt=True,
            use_cuda_graphs=False,
        )
        noise = (
            misc.arch_invariant_rand(
                (sample_count,) + state_shape,
                torch.float32,
                self.pipe.tensor_kwargs["device"],
                seed,
            )
            * self.pipe.scheduler.config.sigma_max
        )
        if self.pipe.dit.is_context_parallel_enabled:
            noise = split_inputs_cp(
                x=noise,
                seq_dim=2,
                cp_group=self.pipe.get_context_parallel_group(),
            )

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(num_sampling_steps, device=noise.device)
        sample = noise.float()
        previous_x0 = None
        for step_index in range(len(scheduler.timesteps)):
            sigma = scheduler.sigmas[step_index].to(sample.device, dtype=torch.float32)
            predicted_x0 = x0_fn(sample, sigma.repeat(sample.shape[0]))
            sample, previous_x0 = scheduler.step(
                x0_pred=predicted_x0,
                i=step_index,
                sample=sample,
                x0_prev=previous_x0,
            )
        final_sigma = scheduler.sigmas[-1].to(sample.device, dtype=torch.float32)
        generated_latent = x0_fn(sample, final_sigma.repeat(sample.shape[0]))

        if self.pipe.dit.is_context_parallel_enabled:
            context_group = self.pipe.get_context_parallel_group()
            context_size = 1 if context_group is None else context_group.size()
            generated_latent = cat_outputs_cp(generated_latent, seq_dim=2, cp_group=context_group)
            if batch.n_views > 1:
                generated_latent = rearrange(
                    generated_latent,
                    "B C (cp V T) H W -> B C (V cp T) H W",
                    cp=context_size,
                    T=self.pipe.config.state_t // context_size,
                )

        generated_video = self.pipe.decode(generated_latent)
        return ((generated_video / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
