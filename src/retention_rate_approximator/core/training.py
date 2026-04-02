from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from retention_rate_approximator.core.data import RetentionDataset
from retention_rate_approximator.core.modeling import ComplexApproximator


@dataclass(frozen=True)
class TrainingPhase:
    optimizer_name: str
    epochs: int
    trend_trainable: bool
    week_trainable: bool
    learning_rate: float


@dataclass(frozen=True)
class TrainingResult:
    model: ComplexApproximator
    loss_history: list[float]
    predicted: Tensor
    predicted_trend: Tensor
    predicted_week: Tensor
    used_day_numbers: Tensor
    used_retention: Tensor
    used_installs: Tensor


def custom_mse_loss(
    y_pred: Tensor,
    y_true: Tensor,
    sample_size_true: Tensor,
    regularizer: callable,
    regularizer_lambda: float,
) -> Tensor:
    stabilized_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
    temp = torch.square(stabilized_pred - y_true) * sample_size_true / (-stabilized_pred * (stabilized_pred - 1)) / torch.sum(sample_size_true)
    return torch.mean(temp) + regularizer() * regularizer_lambda


def train_retention_model(
    dataset: RetentionDataset,
    first_day_of_week: int,
    main_function_type: int | str,
    chain_function_type: int | str,
    connector_type: int | str,
    patches_dates: Sequence[int] | None,
    bad_dates: Sequence[int] | None = None,
    exclude_patch_dates: bool = True,
    week_function_initial_weights: Sequence[float] | None = None,
    main_function_initial_weights: Sequence[float] | float | None = None,
    training_strategy: Sequence[TrainingPhase] | None = None,
    use_custom_loss: bool = True,
    regularizer_lambda: float = 100.0,
    device: torch.device | None = None,
) -> TrainingResult:
    effective_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalized_patches = sorted({int(value) for value in patches_dates or []})
    excluded_dates = {int(value) for value in bad_dates or []}
    if exclude_patch_dates:
        excluded_dates.update(normalized_patches)

    day_numbers = dataset.day_numbers.to(device=effective_device)
    retention = dataset.retention.to(device=effective_device)
    installs = dataset.installs.to(device=effective_device)

    good_indices = [index for index, day in enumerate(day_numbers.detach().cpu().tolist()) if int(day) not in excluded_dates]
    if not good_indices:
        raise ValueError("No training rows remain after filtering bad dates.")

    used_day_numbers = day_numbers[good_indices].detach()
    used_retention = retention[good_indices].detach()
    used_installs = installs[good_indices].detach()

    model = ComplexApproximator(
        first_day_of_week=first_day_of_week,
        patches_dates=normalized_patches,
        main_function_type=main_function_type,
        chain_functions_type=chain_function_type,
        connector_type=connector_type,
        main_function_initial_weights=main_function_initial_weights,
        week_function_initial_weights=week_function_initial_weights,
    )
    model.init_weights_from_train_data(used_day_numbers, used_retention)
    model.to(device=effective_device)
    model.train()

    phases = list(training_strategy or default_training_strategy())
    loss_history: list[float] = []
    for phase in phases:
        optimizer = _create_optimizer(model, phase)
        model.week_function.requires_grad_(phase.week_trainable)
        for chain_function in model.chain_functions:
            chain_function.requires_grad_(phase.trend_trainable)

        for _ in range(phase.epochs):
            if phase.optimizer_name == "LBFGS":
                def closure() -> Tensor:
                    optimizer.zero_grad()
                    predicted = model(used_day_numbers)
                    loss = (
                        custom_mse_loss(predicted, used_retention, used_installs, model.regularize, regularizer_lambda)
                        if use_custom_loss
                        else torch.nn.functional.mse_loss(predicted, used_retention)
                    )
                    loss.backward()
                    return loss.detach()

                step_loss = optimizer.step(closure)
                loss_history.append(float(step_loss.item()))
                continue

            optimizer.zero_grad()
            predicted = model(used_day_numbers)
            loss = (
                custom_mse_loss(predicted, used_retention, used_installs, model.regularize, regularizer_lambda)
                if use_custom_loss
                else torch.nn.functional.mse_loss(predicted, used_retention)
            )
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.detach().item()))

    model.eval()
    with torch.no_grad():
        predicted = model(day_numbers)
        predicted_trend = model.forward_trend_function(day_numbers)
        predicted_week = model.forward_week_function(day_numbers)

    return TrainingResult(
        model=model,
        loss_history=loss_history,
        predicted=predicted.detach().cpu(),
        predicted_trend=predicted_trend.detach().cpu(),
        predicted_week=predicted_week.detach().cpu(),
        used_day_numbers=used_day_numbers.detach().cpu(),
        used_retention=used_retention.detach().cpu(),
        used_installs=used_installs.detach().cpu(),
    )


def default_training_strategy() -> tuple[TrainingPhase, ...]:
    return (
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("LBFGS", 5, True, False, 0.01),
        TrainingPhase("LBFGS", 5, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.01),
        TrainingPhase("Adam", 500, True, False, 0.001),
        TrainingPhase("Adam", 500, True, True, 0.001),
        TrainingPhase("Adam", 500, True, True, 0.001),
    )


def _create_optimizer(model: ComplexApproximator, phase: TrainingPhase) -> torch.optim.Optimizer:
    if phase.optimizer_name == "LBFGS":
        return torch.optim.LBFGS(model.parameters(), lr=phase.learning_rate)
    return torch.optim.Adam(model.parameters(), lr=phase.learning_rate)
