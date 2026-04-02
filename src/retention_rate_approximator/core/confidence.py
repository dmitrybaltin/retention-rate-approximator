from __future__ import annotations

from typing import Literal

import numpy as np
from torch import Tensor

ConfidenceBandMode = Literal['off', '2sigma', '3sigma']


def band_multiplier(mode: ConfidenceBandMode) -> float:
    if mode == '2sigma':
        return 2.0
    if mode == '3sigma':
        return 3.0
    return 0.0


def build_segment_confidence_sigma(
    day_numbers: Tensor,
    installs: Tensor,
    predicted: Tensor,
    patch_dates: list[int] | tuple[int, ...],
    used_day_numbers: Tensor,
) -> np.ndarray:
    days = day_numbers.detach().cpu().numpy().astype(int)
    installs_np = installs.detach().cpu().numpy().astype(float)
    predicted_np = predicted.detach().cpu().numpy().astype(float)
    used_days_np = used_day_numbers.detach().cpu().numpy().astype(int)

    clipped_predicted = np.clip(predicted_np, 1e-6, 1 - 1e-6)
    point_variance = clipped_predicted * (1.0 - clipped_predicted) / np.maximum(installs_np, 1.0)
    segment_indices = np.searchsorted(np.asarray(sorted(int(value) for value in patch_dates), dtype=int), days, side='right')
    used_mask = np.isin(days, used_days_np)

    sigma_by_segment: dict[int, float] = {}
    for segment_index in np.unique(segment_indices):
        segment_mask = segment_indices == segment_index
        segment_used_mask = segment_mask & used_mask
        active_mask = segment_used_mask if np.any(segment_used_mask) else segment_mask
        weighted_variance = float(np.sum(point_variance[active_mask] * installs_np[active_mask]) / np.sum(installs_np[active_mask]))
        sigma_by_segment[int(segment_index)] = float(np.sqrt(max(weighted_variance, 0.0)))

    return np.asarray([sigma_by_segment[int(index)] for index in segment_indices], dtype=float)
