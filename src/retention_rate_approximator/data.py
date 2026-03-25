from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor


@dataclass(frozen=True)
class RetentionDataset:
    day_numbers: Tensor
    installs: Tensor
    retention: Tensor
    retention_mean: Tensor
    first_date: datetime
    frame: pd.DataFrame


def load_retention_csv(path: str | Path) -> RetentionDataset:
    frame = pd.read_csv(path)

    if 'date' in frame.columns and not pd.api.types.is_numeric_dtype(frame['date']):
        frame['date'] = pd.to_datetime(frame['date'])
        dates_list = frame['date'].tolist()
        first_date = min(dates_list)
        day_numbers_series = pd.Series([(current_date - first_date).days for current_date in dates_list], name='day_number')
    elif 'day_number' in frame.columns:
        day_numbers_series = pd.to_numeric(frame['day_number'], errors='raise')
        first_date = datetime(1970, 1, 1)
    elif 'date' in frame.columns:
        day_numbers_series = pd.to_numeric(frame['date'], errors='raise')
        first_date = datetime(1970, 1, 1)
    else:
        raise ValueError("CSV must contain either 'date' or 'day_number' column.")

    return RetentionDataset(
        day_numbers=torch.clamp(torch.tensor(day_numbers_series.values), 0, 100_000_000_000_000).float(),
        installs=torch.clamp(torch.tensor(frame['installs'].values), 1, 10_000_000_000),
        retention=torch.clamp(torch.tensor(frame['retention'].values), 0, 1),
        retention_mean=torch.clamp(torch.tensor(frame['retention_mean'].values), 0, 1),
        first_date=first_date,
        frame=frame,
    )


def save_retention_csv(
    path: str | Path,
    day_numbers: Tensor,
    installs: Tensor,
    retention: Tensor,
    retention_mean: Tensor | None = None,
    release_date: date | datetime | None = None,
) -> Path:
    target = Path(path)
    mean_tensor = torch.zeros_like(installs) if retention_mean is None else retention_mean

    if release_date is None:
        table = torch.stack((day_numbers, installs, retention, mean_tensor), dim=1).detach().cpu().numpy()
        frame = pd.DataFrame(table, columns=['day_number', 'installs', 'retention', 'retention_mean'])
    else:
        date_list = [release_date + timedelta(days=int(day)) for day in day_numbers.detach().cpu().tolist()]
        table = torch.stack((day_numbers, installs, retention, mean_tensor), dim=1).detach().cpu().numpy()
        frame = pd.DataFrame(table, columns=['day_number', 'installs', 'retention', 'retention_mean'])
        frame.insert(0, 'date', date_list)

    frame.to_csv(target, index=False)
    return target