"""
AxialStripCNN — task-specific backbone for boiler wall wear strips.

Designed for 256×256 axial strip patches:
  - Horizontal (1×k) convs capture wear progression along strip width
  - Vertical (k×1) convs capture circumferential texture bands
  - No external pretrained weights required
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AxialConv(nn.Module):
    """Factorized axial convolution: sum of horizontal and vertical responses."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        h_kernel: int = 7,
        v_kernel: int = 5,
    ) -> None:
        super().__init__()
        h_pad = h_kernel // 2
        v_pad = v_kernel // 2
        self.h_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, h_kernel),
            padding=(0, h_pad),
            bias=False,
        )
        self.v_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(v_kernel, 1),
            padding=(v_pad, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.h_conv(x) + self.v_conv(x)))


class AxialStripBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.axial = AxialConv(in_channels, out_channels)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.axial(x)
        out = self.refine(out)
        if self.downsample is not None:
            out = self.downsample(out)
        return out + self.shortcut(x)


class AxialStripStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        blocks = [AxialStripBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(AxialStripBlock(out_channels, out_channels, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class AxialStripCNN(nn.Module):
    """
    Lightweight strip encoder backbone for SOFormer.

    Input:  [B, 3, 256, 256]
    Output: [B, num_features]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 48,
        num_features: int = 512,
        blocks_per_stage: int = 2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = AxialStripStage(c, c * 2, blocks_per_stage, stride=2)
        self.stage2 = AxialStripStage(c * 2, c * 4, blocks_per_stage, stride=2)
        self.stage3 = AxialStripStage(c * 4, c * 8, blocks_per_stage, stride=2)
        self.stage4 = AxialStripStage(c * 8, num_features, blocks_per_stage, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return x.flatten(1)


CUSTOM_BACKBONE_NAMES = frozenset({"axial_strip_cnn", "ascnn"})


def build_axial_strip_cnn(
    backbone: str,
    base_channels: int = 48,
    num_features: int = 512,
    blocks_per_stage: int = 2,
    **_kwargs,
) -> AxialStripCNN:
    if backbone not in CUSTOM_BACKBONE_NAMES:
        raise ValueError(f"Unknown custom backbone: {backbone}")
    return AxialStripCNN(
        base_channels=base_channels,
        num_features=num_features,
        blocks_per_stage=blocks_per_stage,
    )
