from __future__ import annotations

from pathlib import Path
from typing import Final
from uuid import uuid4

from retention_rate_approximator.core.synthetic import generate_retention_dataset
from retention_rate_approximator.core.training import TrainingPhase
from retention_rate_approximator.core.data import save_retention_csv
from retention_rate_approximator.core.plotting import plot_synthetic_dataset
import pandas as pd

ARTIFACTS_DIR: Final[Path] = Path('.artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)


def parse_int_list(raw_value: str) -> list[int]:
    if not raw_value.strip():
        return []
    return [int(item.strip()) for item in raw_value.split(',') if item.strip()]


def parse_float_list(raw_value: str) -> list[float]:
    if not raw_value.strip():
        return []
    return [float(item.strip()) for item in raw_value.split(',') if item.strip()]


def resolve_training_strategy(mode: str) -> list[TrainingPhase]:
    if mode == 'Fast':
        return [
            TrainingPhase('Adam', 150, True, False, 0.01),
            TrainingPhase('Adam', 150, True, True, 0.005),
        ]
    return [
        TrainingPhase('Adam', 300, True, False, 0.01),
        TrainingPhase('LBFGS', 3, True, False, 0.01),
        TrainingPhase('Adam', 300, True, True, 0.001),
    ]


def ensure_session_id(session_id: str | None) -> str:
    if session_id:
        return session_id
    return uuid4().hex


def build_session_artifacts_dir(session_id: str | None) -> tuple[Path, str]:
    active_session_id = ensure_session_id(session_id)
    session_dir = ARTIFACTS_DIR / active_session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, active_session_id


def build_session_download_path(name: str, session_id: str | None) -> tuple[Path, str]:
    source = Path(name)
    safe_stem = ''.join(char if char.isalnum() or char in '-_' else '_' for char in source.stem)
    suffix = source.suffix or '.csv'
    session_dir, active_session_id = build_session_artifacts_dir(session_id)
    return session_dir / f'{safe_stem}{suffix}', active_session_id


def create_generated_frame(
    total_days: int,
    first_day_of_week: int,
    patches_dates: str,
    main_function_type: str,
    chain_function_type: str,
    main_function_weights: str,
    chain_function_weights: str,
    week_function_weights: str,
    daily_installs_mean: int,
    daily_installs_sigma: int,
    session_id: str | None,
) -> tuple[pd.DataFrame, object, str, str, str]:
    patch_values = parse_int_list(patches_dates)
    main_weight_values = parse_float_list(main_function_weights)
    chain_weight_values = parse_float_list(chain_function_weights)
    week_weight_values = parse_float_list(week_function_weights)
    generated = generate_retention_dataset(
        total_days=total_days,
        first_day_of_week=first_day_of_week,
        patches_dates=patch_values,
        main_function_type=main_function_type,
        chains_functions_type=chain_function_type,
        main_function_weights=main_weight_values,
        chains_functions_weights=chain_weight_values,
        week_function_weights=week_weight_values,
        daily_installs_mean=daily_installs_mean,
        daily_installs_sigma=daily_installs_sigma,
    )
    output_path, active_session_id = build_session_download_path('demo_dataset.csv', session_id)
    save_retention_csv(
        path=output_path,
        day_numbers=generated.day_numbers,
        installs=generated.installs,
        retention=generated.retention,
        retention_mean=generated.retention_trend,
    )
    frame = pd.DataFrame(
        {
            'day_number': generated.day_numbers.detach().cpu().numpy(),
            'installs': generated.installs.detach().cpu().numpy(),
            'retention': generated.retention.detach().cpu().numpy(),
            'retention_mean': generated.retention_trend.detach().cpu().numpy(),
        }
    )
    figure = plot_synthetic_dataset(generated)
    details = (
        '### Demo dataset generated\n'
        f'- Days: {total_days}\n'
        f"- Patch dates: {', '.join(str(value) for value in generated.patches_dates) or 'none'}\n"
        f'- Anomaly days: {len(generated.bad_days)}'
    )
    return frame, figure, details, str(output_path), active_session_id
