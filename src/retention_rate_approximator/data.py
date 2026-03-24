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
    frame["date"] = pd.to_datetime(frame["date"])
    dates_list = frame["date"].tolist()
    first_date = min(dates_list)
    day_numbers = [(current_date - first_date).days for current_date in dates_list]
    return RetentionDataset(
        day_numbers=torch.clamp(torch.tensor(day_numbers), 0, 100_000_000_000_000).float(),
        installs=torch.clamp(torch.tensor(frame["installs"].values), 1, 10_000_000_000),
        retention=torch.clamp(torch.tensor(frame["retention"].values), 0, 1),
        retention_mean=torch.clamp(torch.tensor(frame["retention_mean"].values), 0, 1),
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
        frame = pd.DataFrame(table, columns=["date", "installs", "retention", "retention_mean"])
    else:
        date_list = [release_date + timedelta(days=int(day)) for day in day_numbers.detach().cpu().tolist()]
        table = torch.stack((installs, retention, mean_tensor), dim=1).detach().cpu().numpy()
        frame = pd.DataFrame(table, columns=["installs", "retention", "retention_mean"])
        frame.insert(0, "date", date_list)
    frame.to_csv(target, index=False)
    return target
