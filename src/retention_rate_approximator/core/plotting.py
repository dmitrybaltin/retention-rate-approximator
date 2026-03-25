from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from torch import Tensor

from retention_rate_approximator.core.synthetic import GeneratedRetentionDataset
from retention_rate_approximator.core.training import TrainingResult


@dataclass(frozen=True)
class FitPlotData:
    day_numbers: Tensor
    retention: Tensor
    predicted: Tensor
    predicted_trend: Tensor
    used_day_numbers: Tensor
    used_retention: Tensor
    retention_mean: Tensor | None = None


def plot_fit_results(data: FitPlotData) -> Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_all = data.day_numbers.detach().cpu().numpy()
    axis.plot(x_all, data.retention.detach().cpu().numpy(), color='royalblue', label='Source data')
    axis.plot(x_all, data.predicted.detach().cpu().numpy(), color='darkmagenta', linewidth=2.5, label='Predicted data')
    axis.plot(x_all, data.predicted_trend.detach().cpu().numpy(), color='firebrick', linewidth=2.0, label='Predicted trend')
    axis.scatter(
        data.used_day_numbers.detach().cpu().numpy(),
        data.used_retention.detach().cpu().numpy(),
        color='royalblue',
        label='Train data',
        s=20,
    )
    if data.retention_mean is not None:
        axis.plot(x_all, data.retention_mean.detach().cpu().numpy(), color='darkgreen', linestyle='--', label='Reference trend')
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def plot_dataset_preview(frame: pd.DataFrame) -> Figure:
    figure, axis = plt.subplots(figsize=(12, 4))
    x_values = frame['day_number'].to_numpy()
    axis.plot(x_values, frame['retention'].to_numpy(), color='royalblue', label='Retention')
    if 'retention_mean' in frame.columns:
        axis.plot(x_values, frame['retention_mean'].to_numpy(), color='darkgreen', linestyle='--', label='Retention mean')
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.25)
    axis.legend(loc='best')
    figure.tight_layout()
    return figure


def plot_synthetic_dataset(generated: GeneratedRetentionDataset) -> Figure:
    figure, axis = plt.subplots(figsize=(12, 5))
    x_values = generated.day_numbers.detach().cpu().numpy()
    axis.plot(x_values, generated.retention.detach().cpu().numpy(), color='royalblue', label='Synthetic data')
    axis.plot(
        x_values,
        generated.retention_with_oscillations.detach().cpu().numpy(),
        color='darkgreen',
        label='Trend with weekly seasonality',
    )
    axis.plot(x_values, generated.retention_trend.detach().cpu().numpy(), color='firebrick', linewidth=2.0, label='Trend')
    axis.set_xlabel('Day number')
    axis.set_ylabel('Retention')
    axis.set_ylim(bottom=0.0)
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