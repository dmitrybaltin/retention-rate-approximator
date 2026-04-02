from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from torch import Tensor

from retention_rate_approximator.core.confidence import ConfidenceBandMode, band_multiplier
from retention_rate_approximator.core.synthetic import GeneratedRetentionDataset
from retention_rate_approximator.core.training import TrainingResult

YAxisMode = Literal['zero', 'auto']


@dataclass(frozen=True)
class FitPlotData:
    day_numbers: Tensor
    retention: Tensor
    predicted: Tensor
    predicted_trend: Tensor
    used_day_numbers: Tensor
    used_retention: Tensor
    retention_mean: Tensor | None = None


def _apply_y_axis_mode(axis: plt.Axes, series: list[np.ndarray], mode: YAxisMode) -> None:
    if mode == 'zero':
        axis.set_ylim(bottom=0.0)
        return
    flattened = np.concatenate([values.reshape(-1) for values in series if values.size > 0])
    if flattened.size == 0:
        axis.set_ylim(bottom=0.0)
        return
    y_min = float(np.min(flattened))
    y_max = float(np.max(flattened))
    span = max(y_max - y_min, 1e-4)
    padding = span * 0.08
    lower = max(0.0, y_min - padding)
    upper = min(1.0, y_max + padding)
    if upper <= lower:
        upper = min(1.0, lower + 0.05)
    axis.set_ylim(lower, upper)


def plot_fit_results(data: FitPlotData, y_axis_mode: YAxisMode = 'zero') -> Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_all = data.day_numbers.detach().cpu().numpy()
    retention = data.retention.detach().cpu().numpy()
    predicted = data.predicted.detach().cpu().numpy()
    predicted_trend = data.predicted_trend.detach().cpu().numpy()
    used_retention = data.used_retention.detach().cpu().numpy()
    axis.plot(x_all, retention, color='royalblue', label='Source data')
    axis.plot(x_all, predicted, color='darkmagenta', linewidth=2.5, label='Predicted data')
    axis.plot(x_all, predicted_trend, color='firebrick', linewidth=2.0, label='Predicted trend')
    axis.scatter(
        data.used_day_numbers.detach().cpu().numpy(),
        used_retention,
        color='royalblue',
        label='Train data',
        s=20,
    )
    series = [retention, predicted, predicted_trend, used_retention]
    if data.retention_mean is not None:
        retention_mean = data.retention_mean.detach().cpu().numpy()
        axis.plot(x_all, retention_mean, color='darkgreen', linestyle='--', label='Reference trend')
        series.append(retention_mean)
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    _apply_y_axis_mode(axis, series, y_axis_mode)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def plot_dataset_preview(frame: pd.DataFrame, y_axis_mode: YAxisMode = 'zero') -> Figure:
    figure, axis = plt.subplots(figsize=(12, 4))
    x_values = frame['day_number'].to_numpy()
    retention = frame['retention'].to_numpy()
    axis.plot(x_values, retention, color='royalblue', label='Retention')
    series = [retention]
    if 'retention_mean' in frame.columns:
        retention_mean = frame['retention_mean'].to_numpy()
        axis.plot(x_values, retention_mean, color='darkgreen', linestyle='--', label='Retention mean')
        series.append(retention_mean)
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    _apply_y_axis_mode(axis, series, y_axis_mode)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def plot_generated_frame(frame: pd.DataFrame, y_axis_mode: YAxisMode = 'zero') -> Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_values = frame['day_number'].to_numpy()
    retention = frame['retention'].to_numpy()
    series = [retention]
    axis.plot(x_values, retention, color='royalblue', label='Synthetic data')
    if 'retention_with_oscillations' in frame.columns:
        oscillations = frame['retention_with_oscillations'].to_numpy()
        axis.plot(x_values, oscillations, color='darkgreen', label='Trend with weekly seasonality')
        series.append(oscillations)
    if 'retention_mean' in frame.columns:
        trend = frame['retention_mean'].to_numpy()
        axis.plot(x_values, trend, color='firebrick', linewidth=2.0, label='Trend')
        series.append(trend)
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    _apply_y_axis_mode(axis, series, y_axis_mode)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def plot_prediction_chart(
    frame: pd.DataFrame,
    y_axis_mode: YAxisMode = 'zero',
    confidence_band_mode: ConfidenceBandMode = 'off',
) -> Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_values = frame['day_number'].to_numpy()
    retention = frame['retention'].to_numpy()
    predicted_retention = frame['predicted_retention'].to_numpy()
    predicted_trend = frame['predicted_trend'].to_numpy()
    axis.plot(x_values, retention, color='royalblue', label='Source data')
    multiplier = band_multiplier(confidence_band_mode)
    series = [retention, predicted_retention, predicted_trend]
    if multiplier > 0.0 and 'confidence_sigma' in frame.columns:
        confidence_sigma = frame['confidence_sigma'].to_numpy()
        lower = np.clip(predicted_retention - multiplier * confidence_sigma, 0.0, 1.0)
        upper = np.clip(predicted_retention + multiplier * confidence_sigma, 0.0, 1.0)
        axis.fill_between(x_values, lower, upper, color='gray', alpha=0.22, label=f'{int(multiplier)}? confidence band')
        series.extend([lower, upper])
    axis.plot(x_values, predicted_retention, color='darkmagenta', linewidth=2.5, label='Predicted data')
    axis.plot(x_values, predicted_trend, color='firebrick', linewidth=2.0, label='Predicted trend')
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    _apply_y_axis_mode(axis, series, y_axis_mode)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def plot_synthetic_dataset(generated: GeneratedRetentionDataset, y_axis_mode: YAxisMode = 'zero') -> Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_values = generated.day_numbers.detach().cpu().numpy()
    retention = generated.retention.detach().cpu().numpy()
    oscillations = generated.retention_with_oscillations.detach().cpu().numpy()
    trend = generated.retention_trend.detach().cpu().numpy()
    axis.plot(x_values, retention, color='royalblue', label='Synthetic data')
    axis.plot(
        x_values,
        oscillations,
        color='darkgreen',
        label='Trend with weekly seasonality',
    )
    axis.plot(x_values, trend, color='firebrick', linewidth=2.0, label='Trend')
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    _apply_y_axis_mode(axis, [retention, oscillations, trend], y_axis_mode)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def build_prediction_frame(
    day_numbers: Tensor,
    observed_retention: Tensor,
    predicted_retention: Tensor,
    predicted_trend: Tensor,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'day_number': day_numbers.detach().cpu().numpy(),
            'retention': observed_retention.detach().cpu().numpy(),
            'predicted_retention': predicted_retention.detach().cpu().numpy(),
            'predicted_trend': predicted_trend.detach().cpu().numpy(),
        }
    )


def build_training_summary(result: TrainingResult) -> str:
    final_loss = result.loss_history[-1] if result.loss_history else float('nan')
    summary_lines = [
        '### Training summary',
        f'- Steps: {len(result.loss_history)}',
        f'- Final loss: {final_loss:.6f}',
        f'- Train rows: {len(result.used_day_numbers)}',
    ]
    for name, values in result.model.summary_parameters().items():
        formatted_values = ', '.join(f'{value:.6f}' for value in values)
        summary_lines.append(f'- `{name}`: {formatted_values}')
    return '\n'.join(summary_lines)
