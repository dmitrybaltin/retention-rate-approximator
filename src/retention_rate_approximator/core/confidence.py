from __future__ import annotations

from typing import Literal

import numpy as np
from torch import Tensor

ConfidenceBandMode = Literal['off', '1sigma', '2sigma', '3sigma']
ConfidenceTargetMode = Literal['predicted', 'trend']


def band_multiplier(mode: ConfidenceBandMode) -> float:
    if mode == '1sigma':
        return 1.0
    if mode == '2sigma':
        return 2.0
    if mode == '3sigma':
        return 3.0
    return 0.0


def build_segment_confidence_sigma(
    day_numbers: Tensor,
    observed_retention: Tensor,
    installs: Tensor,
    patch_dates: list[int] | tuple[int, ...],
    used_day_numbers: Tensor,
) -> np.ndarray:
    days = day_numbers.detach().cpu().numpy().astype(int)
    retention_np = observed_retention.detach().cpu().numpy().astype(float)
    installs_np = installs.detach().cpu().numpy().astype(float)
    used_days_np = used_day_numbers.detach().cpu().numpy().astype(int)

    segment_indices = np.searchsorted(np.asarray(sorted(int(value) for value in patch_dates), dtype=int), days, side='right')
    used_mask = np.isin(days, used_days_np)

    sigma_by_segment: dict[int, float] = {}
    for segment_index in np.unique(segment_indices):
        segment_mask = segment_indices == segment_index
        segment_used_mask = segment_mask & used_mask
        active_mask = segment_used_mask if np.any(segment_used_mask) else segment_mask
        segment_installs = installs_np[active_mask]
        segment_retention = retention_np[active_mask]
        total_installs = float(np.sum(segment_installs))
        if total_installs <= 0.0:
            sigma_by_segment[int(segment_index)] = 0.0
            continue
        weighted_retention = float(np.sum(segment_retention * segment_installs) / total_installs)
        weighted_retention = float(np.clip(weighted_retention, 1e-6, 1 - 1e-6))
        sigma_by_segment[int(segment_index)] = float(np.sqrt(weighted_retention * (1.0 - weighted_retention) / total_installs))

    return np.asarray([sigma_by_segment[int(index)] for index in segment_indices], dtype=float)
