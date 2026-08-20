"""Confidence-guided deformable BEV enhancement (L3 of HC-MVMM).

Implements Eqs. 19-26 of the paper. A lightweight deformable-attention
branch produces a residual enhancement of the CNN BEV feature; the
sampling-offset scale is calibrated by both the local feature spread
(``s_spread``) and the BEV confidence map ``r_bev`` propagated from L2.
A spatially gated residual connection determines how much enhancement to
inject at each BEV location.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_bev_backbone import BaseBEVBackbone


class LightDeformableAttention(nn.Module):
    """Lightweight deformable attention with feature-spread / confidence gating.

    Output is initialised to zero so that the enhanced branch starts as a
    no-op and the model behaves like the CNN baseline at the beginning of
    training. As training progresses the residual is gradually opened up by
    the spatial gate (see :class:`EnhancedBEVBackbone`).
    """

    def __init__(self, d_model=64, n_heads=4, n_points=3,
                 local_kernel=3, surround_kernel=11,
                 use_spread_guidance=True, use_conf_guidance=True):
        """Initialises the deformable attention parameters.

        Args:
            d_model: Internal embedding dimension.
            n_heads: Number of attention heads.
            n_points: Number of sampling locations per head per query.
            local_kernel: Kernel size for the local pooling used in ``s_spread``.
            surround_kernel: Kernel size for the surround pooling in ``s_spread``.
            use_spread_guidance: Enable the feature-spread offset scale.
            use_conf_guidance: Multiply the offset scale by ``1.5 - r_bev``.
        """
        super().__init__()
        assert d_model % n_heads == 0, (
            f'd_model({d_model}) must be divisible by n_heads({n_heads})'
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.local_kernel = local_kernel
        self.surround_kernel = surround_kernel
        self.use_spread_guidance = use_spread_guidance
        self.use_conf_guidance = use_conf_guidance

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()
        # Buffer used by :class:`EnhancedBEVBackbone` debug print.
        self._last_offset_scale = None

    def _reset_parameters(self):
        # Sampling offsets are initialised on equally spaced rays around the
        # reference point, scaled by the sampling-point index.
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = (
            torch.arange(self.n_heads, dtype=torch.float32)
            * (2.0 * math.pi / self.n_heads)
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        ).view(self.n_heads, 1, 2).repeat(1, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, i, :] *= (i + 1)
        self.sampling_offsets.bias.data = grid_init.view(-1)

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        # Zero-init the output projection so the residual starts at zero.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _compute_offset_scale(self, bev_feat):
        """Computes the feature-spread offset scale ``s_spread`` (Eq. 21).

        Args:
            bev_feat: ``[B, C_orig, H, W]`` CNN BEV feature.

        Returns:
            ``[B, 1, H, W]`` offset scale in ``[0.05, 1.0]``.
        """
        feat_mag = bev_feat.norm(dim=1, keepdim=True)
        lk = self.local_kernel
        sk = self.surround_kernel
        local = F.avg_pool2d(feat_mag, lk, stride=1, padding=lk // 2)
        surround = F.avg_pool2d(feat_mag, sk, stride=1, padding=sk // 2)
        return (surround / (local + 1e-6)).clamp(min=0.05, max=1.0)

    def forward(self, x, bev_feat_for_scale=None, bev_conf=None):
        """Runs deformable attention with confidence-modulated sampling.

        Args:
            x: ``[B, C, H, W]`` low-rank BEV feature obtained by projection.
            bev_feat_for_scale: ``[B, C_orig, H, W]`` CNN BEV feature, used
                to compute ``s_spread``.
            bev_conf: ``[B, 1, H, W]`` BEV confidence map from L2.

        Returns:
            ``[B, C, H, W]`` enhanced BEV feature.
        """
        B, C, H, W = x.shape
        N = H * W

        x_flat = x.flatten(2).permute(0, 2, 1)

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=x.device),
            torch.linspace(0.5, W - 0.5, W, device=x.device),
            indexing='ij',
        )
        ref = torch.stack([ref_x.reshape(-1) / W, ref_y.reshape(-1) / H], dim=-1)
        ref = ref[None].expand(B, -1, -1)

        offsets = self.sampling_offsets(x_flat).view(
            B, N, self.n_heads, self.n_points, 2
        )

        # ``s_offset`` from Eq. 22: combine feature spread and BEV confidence.
        offset_scale = None
        if self.use_spread_guidance and bev_feat_for_scale is not None:
            offset_scale = self._compute_offset_scale(bev_feat_for_scale)
        elif self.use_conf_guidance and bev_conf is not None:
            offset_scale = torch.ones_like(bev_conf)

        if offset_scale is not None and self.use_conf_guidance and bev_conf is not None:
            offset_scale = offset_scale * (1.5 - bev_conf).clamp(0.5, 1.5)

        if offset_scale is not None:
            scale = offset_scale.flatten(2).permute(0, 2, 1).view(B, N, 1, 1, 1)
            offsets = offsets * scale
            self._last_offset_scale = offset_scale
        else:
            self._last_offset_scale = torch.ones(
                (B, 1, H, W), dtype=x.dtype, device=x.device
            )

        # Normalise pixel-space offsets to ``[0, 1]``.
        offsets = offsets / torch.tensor(
            [W, H], device=x.device, dtype=x.dtype
        ).view(1, 1, 1, 1, 2)
        sample_locs = (ref[:, :, None, None, :] + offsets).clamp(0, 1)

        attn_w = self.attention_weights(x_flat).view(
            B, N, self.n_heads, self.n_points
        )
        attn_w = F.softmax(attn_w, dim=-1)

        value = self.value_proj(x_flat)
        value_2d = value.permute(0, 2, 1).view(B, C, H, W)

        sampled_list = []
        for h in range(self.n_heads):
            head_val = value_2d[:, h * self.head_dim:(h + 1) * self.head_dim, :, :]
            grid = sample_locs[:, :, h, :, :] * 2 - 1
            # ``grid_sample`` expects (x, y) ordering, but the helper above
            # builds (x, y); flip to follow PyTorch convention.
            grid = grid.flip(-1)
            sampled = F.grid_sample(
                head_val, grid, mode='bilinear',
                padding_mode='zeros', align_corners=False,
            )
            sampled_list.append(sampled.permute(0, 2, 3, 1))

        sampled = torch.stack(sampled_list, dim=2)
        out = (attn_w[..., None] * sampled).sum(dim=3)
        out = out.flatten(2)
        out = self.output_proj(out)
        out = self.norm(x_flat + out)
        return out.permute(0, 2, 1).view(B, C, H, W)


class EnhancedBEVBackbone(nn.Module):
    """``BaseBEVBackbone`` + a residual deformable enhancement branch (L3).

    The enhancement is gated by a small spatial CNN with dilated convolutions
    so the residual can be opened only where it is useful. The gate is
    initialised so the residual is nearly zero at the start of training.
    """

    def __init__(self, cfg, in_channels):
        """Initialises the CNN backbone and the optional enhancement branch.

        Args:
            cfg: ``bev_backbone`` sub-dictionary of the YAML config. The
                ``deformable_enhance`` sub-block controls L3.
            in_channels: Input channel count from L2.
        """
        super().__init__()
        self.base_backbone = BaseBEVBackbone(cfg, in_channels)
        num_features = self.base_backbone.num_bev_features
        self.num_bev_features = num_features

        enhance_cfg = cfg.get('deformable_enhance', {})
        self.use_enhance = enhance_cfg.get('enabled', True)

        if not self.use_enhance:
            self.debug = False
            self.debug_counter = 0
            return

        enhance_dim = enhance_cfg.get('enhance_dim', 64)
        n_heads = enhance_cfg.get('n_heads', 4)
        n_points = enhance_cfg.get('n_points', 3)
        local_kernel = enhance_cfg.get('local_kernel', 3)
        surround_kernel = enhance_cfg.get('surround_kernel', 11)
        gate_hidden_dim = enhance_cfg.get('gate_hidden_dim', 32)
        gate_dilation = enhance_cfg.get('gate_dilation', 3)

        # Channel projection down to ``enhance_dim`` for the attention branch.
        self.proj_down = nn.Sequential(
            nn.Conv2d(num_features, enhance_dim, 1, bias=False),
            nn.BatchNorm2d(enhance_dim),
            nn.ReLU(inplace=True),
        )
        self.deform_attn = LightDeformableAttention(
            d_model=enhance_dim,
            n_heads=n_heads,
            n_points=n_points,
            local_kernel=local_kernel,
            surround_kernel=surround_kernel,
            use_spread_guidance=enhance_cfg.get('use_spread_guidance', True),
            use_conf_guidance=enhance_cfg.get('use_conf_guidance', True),
        )
        self.proj_up = nn.Sequential(
            nn.Conv2d(enhance_dim, num_features, 1, bias=False),
            nn.BatchNorm2d(num_features),
        )

        # Spatial gate ``g(p)`` in Eq. 26. Last layer is zero-init with bias
        # -3 so that ``sigmoid(-3) ~= 0.047`` keeps the residual quiet early.
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(num_features, gate_hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden_dim, gate_hidden_dim, 3,
                      padding=gate_dilation, dilation=gate_dilation, bias=False),
            nn.BatchNorm2d(gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden_dim, 1, 1),
        )
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.constant_(self.spatial_gate[-1].bias, -3.0)

        self.debug = enhance_cfg.get('debug', False)
        self.debug_counter = 0

    def forward(self, batch_dict):
        """Runs the CNN backbone, then conditionally adds the deformable branch."""
        batch_dict = self.base_backbone(batch_dict)

        if not self.use_enhance:
            return batch_dict

        bev_feat = batch_dict['bev_features']
        bev_conf = batch_dict.get('bev_confidence', None)
        if (bev_conf is not None
                and bev_conf.shape[2:] != bev_feat.shape[2:]):
            bev_conf = F.interpolate(
                bev_conf, size=bev_feat.shape[2:],
                mode='bilinear', align_corners=False,
            )

        x = self.proj_down(bev_feat)
        x = self.deform_attn(
            x, bev_feat_for_scale=bev_feat, bev_conf=bev_conf
        )
        x = self.proj_up(x)

        gate_map = self.spatial_gate(bev_feat).sigmoid()
        batch_dict['bev_features'] = bev_feat + gate_map * x

        if self.debug and self.training and self.debug_counter % 200 == 0:
            offset_scale = self.deform_attn._last_offset_scale
            active_ratio = (gate_map > 0.1).float().mean().item()
            print(
                f'\n[EnhancedBEV Debug] Step {self.debug_counter} '
                f'(gate_mode=spatial)', flush=True,
            )
            print(
                f'  Spatial gate - mean: {gate_map.mean().item():.4f}, '
                f'std: {gate_map.std().item():.4f}, '
                f'max: {gate_map.max().item():.4f}, '
                f'min: {gate_map.min().item():.4f}, '
                f'active(>0.1): {active_ratio * 100:.1f}%', flush=True,
            )
            if offset_scale is not None:
                print(
                    f'  Offset scale - mean: {offset_scale.mean().item():.4f}, '
                    f'std: {offset_scale.std().item():.4f}', flush=True,
                )
            print(
                f'  CNN feat norm: {bev_feat.norm().item():.2f}, '
                f'Enhanced norm: {x.norm().item():.2f}', flush=True,
            )
        self.debug_counter += 1
        return batch_dict
