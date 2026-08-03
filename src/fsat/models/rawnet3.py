"""RawNet3 backbone and the binary deepfake detector built on it.

RawNet3 (Jung et al., "Pushing the limits of raw waveform speaker
recognition", arXiv:2203.08488) is the detector F-SAT trains and the strongest
baseline in Table 2 of arXiv:2411.00121v1. As Section 2 of that paper
describes it, RawNet3 "begins by using parameterized filterbanks to extract a
time-frequency representation from the raw waveform and then is followed by
three backbone blocks with residual connections".

This is a dependency-free reimplementation of the reference architecture:

* :class:`ParamSincFB` -- the analytic parameterized sinc filterbank
  (Zeghidour et al., 2018), reimplemented so ``asteroid-filterbanks`` is not
  required. Each of ``n_filters // 2`` band-pass filters contributes a cosine
  and a sine kernel, giving ``n_filters`` output channels.
* :class:`AFMS` -- attentive feature-map scaling.
* :class:`Bottle2neck` -- Res2Net block with dilation, residual connection
  and AFMS, the ECAPA-TDNN-derived backbone block.
* :class:`RawNet3` -- filterbank, three blocks, multi-layer feature
  aggregation, attentive statistics pooling with context.
* :class:`RawNet3Detector` -- the trunk plus a linear real/fake head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================== #
# Front end
# ====================================================================== #
class PreEmphasis(nn.Module):
    """First-order pre-emphasis filter ``y[n] = x[n] - coef * x[n-1]``."""

    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        self.register_buffer(
            "kernel", torch.tensor([[[-coef, 1.0]]], dtype=torch.float32), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), mode="reflect")
        return F.conv1d(x, self.kernel.to(x.dtype))


def _hz_to_mel(hz):
    return 2595.0 * torch.log10(1.0 + torch.as_tensor(hz, dtype=torch.float32) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (torch.as_tensor(mel, dtype=torch.float32) / 2595.0) - 1.0)


class ParamSincFB(nn.Module):
    """Analytic parameterized sinc filterbank with learnable band edges.

    Only the low cutoff and the bandwidth of each filter are learned, so the
    front end stays interpretable while remaining trainable. Filters are
    mel-spaced at initialisation.

    Output has ``n_filters`` channels: ``n_filters // 2`` cosine kernels
    followed by the matching ``n_filters // 2`` sine kernels.
    """

    def __init__(
        self,
        n_filters: int = 256,
        kernel_size: int = 251,
        stride: int = 10,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ) -> None:
        super().__init__()
        if n_filters % 2 != 0:
            raise ValueError(f"n_filters must be even (cos/sin pairs), got {n_filters}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz
        self.cutoff = n_filters // 2

        # Mel-spaced initial band edges.
        low_hz = 30.0
        high_hz = sample_rate / 2.0 - (min_low_hz + min_band_hz)
        mel = torch.linspace(_hz_to_mel(low_hz), _hz_to_mel(high_hz), self.cutoff + 1)
        hz = _mel_to_hz(mel)
        self.low_hz_ = nn.Parameter(hz[:-1].view(-1, 1).clone())
        self.band_hz_ = nn.Parameter(hz.diff().view(-1, 1).clone())

        half_kernel = kernel_size // 2
        n_lin = torch.linspace(0, half_kernel - 1, steps=half_kernel)
        self.register_buffer(
            "window_", (0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / kernel_size)), persistent=False
        )
        n = (kernel_size - 1) / 2.0
        self.register_buffer(
            "n_", (2 * math.pi * torch.arange(-n, 0).view(1, -1) / sample_rate), persistent=False
        )

    def make_filters(self) -> torch.Tensor:
        """Build the ``(n_filters, 1, kernel_size)`` kernel bank from the band edges."""
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2.0,
        )
        band = (high - low)[:, 0]
        n_ = self.n_.to(low.dtype)
        window = self.window_.to(low.dtype)

        ft_low = torch.matmul(low, n_)
        ft_high = torch.matmul(high, n_)
        denom = n_ / 2.0
        scale = 2.0 * band[:, None]

        # Cosine (real) branch.
        bp_left = ((torch.sin(ft_high) - torch.sin(ft_low)) / denom) * window
        bp_center = 2.0 * band.view(-1, 1)
        bp_right = torch.flip(bp_left, dims=[1])
        cos_filters = torch.cat([bp_left, bp_center, bp_right], dim=1) / scale

        # Sine (imaginary) branch, giving the analytic pair.
        bps_left = ((torch.cos(ft_low) - torch.cos(ft_high)) / denom) * window
        bps_center = torch.zeros_like(bp_center)
        bps_right = -torch.flip(bps_left, dims=[1])
        sin_filters = torch.cat([bps_left, bps_center, bps_right], dim=1) / scale

        filters = torch.cat([cos_filters, sin_filters], dim=0)
        return filters.view(self.n_filters, 1, self.kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.conv1d(x, self.make_filters().to(x.dtype), stride=self.stride, padding=0)


# ====================================================================== #
# Backbone blocks
# ====================================================================== #
class AFMS(nn.Module):
    """Attentive feature-map scaling: a per-channel gate from pooled statistics."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels, 1))
        self.fc = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        gate = torch.sigmoid(self.fc(gate)).view(x.size(0), x.size(1), 1)
        return (x + self.alpha) * gate


class Bottle2neck(nn.Module):
    """Res2Net bottleneck with dilated convolutions, residual path and AFMS.

    The ``scale`` hierarchical splits give multiple effective receptive fields
    inside one block, which is what lets three blocks cover a long context.
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        dilation: int = 1,
        scale: int = 8,
        pool: Optional[int] = None,
    ) -> None:
        super().__init__()
        width = int(math.floor(planes / scale))
        self.width = width
        self.nums = scale - 1

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        padding = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=padding)
            for _ in range(self.nums)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(self.nums))

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(pool) if pool else None
        self.afms = AFMS(planes)
        self.residual = (
            nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
            if inplanes != planes
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.bn1(self.relu(self.conv1(x)))
        splits = torch.split(out, self.width, dim=1)

        sp = splits[0]
        for i in range(self.nums):
            if i > 0:
                sp = sp + splits[i]
            sp = self.bns[i](self.relu(self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), dim=1)
        out = torch.cat((out, splits[self.nums]), dim=1)

        out = self.bn3(self.relu(self.conv3(out)))
        out = out + residual
        if self.mp is not None:
            out = self.mp(out)
        return self.afms(out)


# ====================================================================== #
# RawNet3
# ====================================================================== #
class RawNet3(nn.Module):
    """RawNet3 trunk producing a fixed-dimensional utterance embedding."""

    def __init__(
        self,
        out_dim: int = 256,
        channels: int = 1024,
        model_scale: int = 8,
        sinc_filters: int = 256,
        sinc_kernel: int = 251,
        sinc_stride: int = 10,
        sample_rate: int = 16000,
        context: bool = True,
        summed: bool = True,
        log_sinc: bool = True,
        norm_sinc: str = "mean",
    ) -> None:
        super().__init__()
        self.context = context
        self.summed = summed
        self.log_sinc = log_sinc
        self.norm_sinc = norm_sinc
        self.sinc_stride = sinc_stride
        self.sinc_kernel = sinc_kernel

        self.preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.conv1 = ParamSincFB(
            n_filters=sinc_filters,
            kernel_size=sinc_kernel,
            stride=sinc_stride,
            sample_rate=sample_rate,
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(sinc_filters)

        self.layer1 = Bottle2neck(sinc_filters, channels, 3, dilation=2, scale=model_scale, pool=5)
        self.layer2 = Bottle2neck(channels, channels, 3, dilation=3, scale=model_scale, pool=3)
        self.layer3 = Bottle2neck(channels, channels, 3, dilation=4, scale=model_scale)
        self.layer4 = nn.Conv1d(3 * channels, 1536, kernel_size=1)
        self.mp3 = nn.MaxPool1d(3)

        attn_in = 1536 * 3 if context else 1536
        self.attention = nn.Sequential(
            nn.Conv1d(attn_in, 128, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Conv1d(128, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, out_dim)
        self.bn6 = nn.BatchNorm1d(out_dim)
        self.out_dim = out_dim

    @property
    def min_input_samples(self) -> int:
        """Shortest waveform that survives the stride/pooling stack.

        The front end decimates by ``sinc_stride``, then layer1 pools by 5 and
        layer2 by 3, a total factor of 15. Requiring at least 4 surviving
        frames keeps the pooled variance in the attention head well defined.
        """
        min_frames = 15 * 4
        return self.sinc_kernel + self.sinc_stride * (min_frames - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.size(-1) < self.min_input_samples:
            raise ValueError(
                f"input of {x.size(-1)} samples is too short for RawNet3; "
                f"need at least {self.min_input_samples}"
            )

        x = self.preprocess(x)
        x = torch.abs(self.conv1(x))
        if self.log_sinc:
            x = torch.log(x + 1e-6)
        if self.norm_sinc == "mean":
            x = x - torch.mean(x, dim=-1, keepdim=True)
        elif self.norm_sinc == "mean_std":
            std = torch.std(x, dim=-1, keepdim=True).clamp_min(1e-3)
            x = (x - torch.mean(x, dim=-1, keepdim=True)) / std

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        # layer1 pools by 5 and layer2 by 3, so x1 must be pooled by a further 3
        # before it can be summed with x2. mp3 serves that role here and again
        # in the multi-layer feature aggregation below.
        x3 = self.layer3(self.mp3(x1) + x2) if self.summed else self.layer3(x2)

        x = self.layer4(torch.cat((self.mp3(x1), x2, x3), dim=1))
        x = self.relu(x)

        t = x.size(-1)
        if self.context:
            mean = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, t)
            std = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4, max=1e4)).repeat(1, 1, t)
            global_x = torch.cat((x, mean, std), dim=1)
        else:
            global_x = x

        w = self.attention(global_x)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2) * w, dim=2) - mu**2).clamp(min=1e-4, max=1e4))

        x = torch.cat((mu, sg), dim=1)
        x = self.bn5(x)
        x = self.fc6(x)
        return self.bn6(x)


class RawNet3Detector(nn.Module):
    """RawNet3 trunk plus a linear real/fake head.

    Takes a waveform ``(B, T)`` or ``(B, 1, T)`` and returns logits ``(B, 2)``.
    Class index 0 is real (bonafide), 1 is fake (spoof).
    """

    def __init__(self, num_classes: int = 2, embed_dim: int = 256, **backbone_kwargs) -> None:
        super().__init__()
        self.backbone = RawNet3(out_dim=embed_dim, **backbone_kwargs)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    @property
    def min_input_samples(self) -> int:
        return self.backbone.min_input_samples

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(F.relu(self.backbone(x)))


__all__ = [
    "PreEmphasis",
    "ParamSincFB",
    "AFMS",
    "Bottle2neck",
    "RawNet3",
    "RawNet3Detector",
]
