from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from retention_rate_approximator.modeling import ComplexApproximator


@dataclass(frozen=True)
class GeneratedRetentionDataset:
    day_numbers: Tensor
    installs: Tensor
    retention: Tensor
    retention_trend: Tensor
    retention_with_oscillations: Tensor
    patches_dates: list[int]
    bad_days: list[int]
    week_weights: list[float]
    modeled_chains: list[Tensor]
    model: ComplexApproximator


def generate_retention_dataset(
    total_days: int,
    first_day_of_week: int,
    patches_dates: Sequence[int],
    main_function_type: int | str,
    chains_functions_type: int | str,
    main_function_weights: Sequence[float],
    chains_functions_weights: Sequence[Sequence[float] | float],
    week_function_weights: Sequence[float],
    daily_installs_mean: float,
    daily_installs_sigma: float,
) -> GeneratedRetentionDataset:
    x = torch.arange(total_days, dtype=torch.float32)
    trend_model = ComplexApproximator(
        first_day_of_week=first_day_of_week,
        patches_dates=patches_dates,
        main_function_type=main_function_type,
        chain_functions_type=chains_functions_type,
        main_function_initial_weights=main_function_weights,
        chain_functions_initial_weights=chains_functions_weights,
    )
    retention_trend = trend_model(x)

    full_model = ComplexApproximator(
        first_day_of_week=first_day_of_week,
        patches_dates=patches_dates,
        main_function_type=main_function_type,
        chain_functions_type=chains_functions_type,
        main_function_initial_weights=main_function_weights,
        chain_functions_initial_weights=chains_functions_weights,
        week_function_initial_weights=week_function_weights,
    )
    retention_with_oscillations = torch.clamp(full_model(x), 0, 1)
    modeled_chains = [
        (x >= full_model.patches_dates[index]) * chain(x)
        for index, chain in enumerate(full_model.chain_functions)
    ]
    installs = torch.rand_like(retention_with_oscillations) * daily_installs_sigma + daily_installs_mean
    sigma = torch.sqrt(-retention_with_oscillations * (retention_with_oscillations - 1) / installs)
    retention = torch.normal(mean=retention_with_oscillations, std=sigma)
    probability_of_anomaly = 0.3
    chosen = np.random.choice(a=[True, False], size=(total_days), p=[probability_of_anomaly, 1 - probability_of_anomaly])
    bad_days = [index for index, is_anomaly in enumerate(chosen) if bool(is_anomaly)]

    return GeneratedRetentionDataset(
        day_numbers=x,
        installs=installs,
        retention=retention,
        retention_trend=retention_trend,
        retention_with_oscillations=retention_with_oscillations,
        patches_dates=[int(value) for value in patches_dates],
        bad_days=bad_days,
        week_weights=[float(value) for value in week_function_weights],
        modeled_chains=modeled_chains,
        model=full_model,
    )
