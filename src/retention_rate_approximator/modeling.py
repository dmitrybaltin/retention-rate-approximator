from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _to_parameter(values: Sequence[float]) -> nn.Parameter:
    return nn.Parameter(torch.tensor(list(values), dtype=torch.float32))


class ConstantFunction(nn.Module):
    def __init__(self, initial_weights: float | None = None) -> None:
        super().__init__()
        initial_value = 0.25 if initial_weights is None else float(initial_weights)
        self.w = nn.Parameter(torch.tensor([initial_value], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return torch.ones_like(x) * self.w[0]

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.0] if new_w is None else list(new_w)
        self.w = _to_parameter(values)

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        del x_train
        self.w = nn.Parameter(torch.mean(y_train).unsqueeze(0))


class LinearFunction(nn.Module):
    def __init__(self, initial_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        values = [0.25, 0.25] if initial_weights is None else initial_weights
        self.w = _to_parameter(values)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.w[1] + self.w[0]

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.0, 0.0] if new_w is None else list(new_w)
        self.w = _to_parameter(values)

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        first_value = y_train[torch.argmin(x_train)]
        last_value = y_train[torch.argmax(x_train)]
        self.w = _to_parameter([float(last_value), float(first_value - last_value)])


class InverseFunction(nn.Module):
    def __init__(self, initial_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        values = [0.25, 0.25, 1.0] if initial_weights is None else initial_weights
        self.w = _to_parameter(values)

    def forward(self, x: Tensor) -> Tensor:
        return self.w[0] + torch.ones_like(x) * self.w[1] / (x + self.w[2])

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.0, 0.0, 10.0] if new_w is None else list(new_w)
        self.w = _to_parameter(values)

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        first_value = y_train[torch.argmin(x_train)]
        last_value = y_train[torch.argmax(x_train)]
        self.w = _to_parameter([float(last_value), float(first_value - last_value), 1.0])


class InverseFunction4(nn.Module):
    def __init__(self, initial_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        values = [0.25, 0.25, 1.0, 1.0] if initial_weights is None else initial_weights
        self.w = _to_parameter(values)

    def forward(self, x: Tensor) -> Tensor:
        return self.w[0] + torch.ones_like(x) * self.w[1] / (torch.pow(x, self.w[3]) + self.w[2])

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.0, 0.0, 10.0, 1.0] if new_w is None else list(new_w)
        self.w = _to_parameter(values)

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        first_value = y_train[torch.argmin(x_train)]
        last_value = y_train[torch.argmax(x_train)]
        self.w = _to_parameter([float(last_value), float(first_value - last_value), 1.0, 1.0])


class LinearFractionalFunctionNew(nn.Module):
    def __init__(self, initial_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        values = [0.5, 0.25, 0.1] if initial_weights is None else list(initial_weights)
        self.w0 = nn.Parameter(torch.tensor([values[0]], dtype=torch.float32))
        self.w1 = nn.Parameter(torch.tensor([values[1]], dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor([values[2]], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return (x / (x + 1 / self.w2[0])) * (self.w1[0] - self.w0[0]) + self.w0[0]

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.5, 0.25, 0.1] if new_w is None else list(new_w)
        self.w0 = nn.Parameter(torch.tensor([values[0]], dtype=torch.float32))
        self.w1 = nn.Parameter(torch.tensor([values[1]], dtype=torch.float32))
        self.w2 = nn.Parameter(torch.tensor([values[2]], dtype=torch.float32))

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        first_value = torch.max(y_train)
        last_value = torch.min(y_train)
        w0 = max(float(first_value), 0.1)
        w1 = max(float(first_value - last_value), 0.0)
        w1 = min(w1, w0)
        x_mean = torch.mean(x_train)
        y_mean = torch.mean(y_train)
        if float(x_mean) != 0.0:
            w2 = float((w1 / (w0 - float(y_mean)) - 1) / x_mean)
        else:
            w2 = 0.05
        self.reset_weights([w0, w1, w2])


class LinearFractionalFunction(nn.Module):
    def __init__(self, initial_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        values = [0.5, 0.25, 20.0] if initial_weights is None else initial_weights
        self.w = _to_parameter(values)

    def forward(self, x: Tensor) -> Tensor:
        return (x / (x + self.w[2])) * (-self.w[1]) + self.w[0]

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.0, 0.0, 10.0] if new_w is None else list(new_w)
        self.w = _to_parameter(values)

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        first_value = torch.max(y_train)
        last_value = torch.min(y_train)
        w0 = max(float(first_value), 0.1)
        w1 = max(float(first_value - last_value), 0.0)
        w1 = min(w1, w0)
        x_mean = torch.mean(x_train)
        y_mean = torch.mean(y_train)
        if float(y_mean) - w0 != 0.0:
            w2 = float(x_mean * (w1 / (w0 - float(y_mean)) - 1))
        else:
            w2 = 20.0
        self.reset_weights([w0, w1, w2])


class SigmaFunction(nn.Module):
    def __init__(self, initial_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        values = [0.5, 0.25, 20.0] if initial_weights is None else initial_weights
        self.w = _to_parameter(values)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(-x * self.w[2] * 2) * self.w[1] + self.w[0]

    def reset_weights(self, new_w: Sequence[float] | None = None) -> None:
        values = [0.0, 0.0, 10.0] if new_w is None else list(new_w)
        self.w = _to_parameter(values)

    def init_weights_from_train_data(self, x_train: Tensor, y_train: Tensor) -> None:
        first_value = torch.max(y_train)
        last_value = torch.min(y_train)
        w0 = float(first_value)
        w1 = float(first_value - last_value)
        x_mean = torch.mean(x_train)
        y_mean = torch.mean(y_train)
        if float(x_mean) != 0.0:
            w2 = float((w1 / (w0 - float(y_mean)) - 1) / x_mean)
        else:
            w2 = 0.05
        self.reset_weights([w0, w1, w2])


class WeekFunction(nn.Module):
    def __init__(
        self,
        first_day_of_week: int,
        regularizer_base: int,
        initial_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.first_day_of_week = first_day_of_week
        values = torch.ones(7, dtype=torch.float32) if initial_weights is None else torch.tensor(list(initial_weights), dtype=torch.float32)
        self.w = nn.Parameter(values)
        self.regularizer_base = regularizer_base

    def forward(self, day_numbers: Tensor) -> Tensor:
        day_of_week = (day_numbers.to(dtype=torch.long) + 7 - self.first_day_of_week) % 7
        return self.w[day_of_week]

    def regularize(self) -> Tensor:
        return torch.square(torch.mean(self.w) - self.regularizer_base)


def multiply_connector(input1: Tensor, input2: Tensor) -> Tensor:
    return torch.mul(input1, input2)


def additive_connector(input1: Tensor, input2: Tensor) -> Tensor:
    return torch.add(input1, input2)


TrendModule = ConstantFunction | LinearFunction | InverseFunction | LinearFractionalFunction | LinearFractionalFunctionNew | InverseFunction4 | SigmaFunction
ChainModule = ConstantFunction | LinearFunction
ConnectorFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    constructor: type[nn.Module]
    default_weights: tuple[float, ...]


class ApproximatorsFactory:
    main_functions: tuple[FunctionSpec, ...] = (
        FunctionSpec("w0", ConstantFunction, (0.25,)),
        FunctionSpec("w0+w1*x", LinearFunction, (0.25, 0.0)),
        FunctionSpec("w0+w1/(w2+x)", InverseFunction, (0.25, 0.25, 1.0)),
        FunctionSpec("w0-w1*x/(w2+x)", LinearFractionalFunction, (0.5, 0.25, 10.0)),
        FunctionSpec("w0-(w0-w1)*x/(1/w2+x)", LinearFractionalFunctionNew, (0.5, 0.25, 0.1)),
        FunctionSpec("w0+w1/(w2+pow(x,w3))", InverseFunction4, (0.25, 0.25, 1.0, 1.0)),
        FunctionSpec("w0+w1*Sigmoid(x*w3)", SigmaFunction, (0.5, 0.25, 0.05)),
    )
    chain_functions: tuple[FunctionSpec, ...] = (
        FunctionSpec("w0", ConstantFunction, (0.25,)),
        FunctionSpec("w0+w1*x", LinearFunction, (0.25, 0.0)),
    )
    connectors: tuple[tuple[str, ConnectorFn, int], ...] = (
        ("mul", multiply_connector, 1),
        ("add", additive_connector, 0),
    )

    @staticmethod
    def create_main_function(
        function_type: int | str | None,
        initial_weights: Sequence[float] | float | None,
    ) -> nn.Module:
        spec = ApproximatorsFactory._find_function_spec(ApproximatorsFactory.main_functions, function_type)
        return spec.constructor(initial_weights)

    @staticmethod
    def create_chain_function(
        function_type: int | str | None,
        initial_weights: Sequence[float] | float | None,
    ) -> nn.Module:
        spec = ApproximatorsFactory._find_function_spec(ApproximatorsFactory.chain_functions, function_type)
        return spec.constructor(initial_weights)

    @staticmethod
    def create_connector(connector_type: int | str | None) -> tuple[ConnectorFn, int]:
        if connector_type is None:
            return ApproximatorsFactory.connectors[0][1], ApproximatorsFactory.connectors[0][2]
        for index, row in enumerate(ApproximatorsFactory.connectors):
            if connector_type == index or connector_type == str(index) or connector_type == row[0]:
                return row[1], row[2]
        return ApproximatorsFactory.connectors[0][1], ApproximatorsFactory.connectors[0][2]

    @staticmethod
    def get_main_function_weights_number(function_type: int | str | None) -> tuple[float, ...] | None:
        if function_type is None:
            return None
        spec = ApproximatorsFactory._find_function_spec(ApproximatorsFactory.main_functions, function_type)
        return spec.default_weights

    @staticmethod
    def _find_function_spec(specs: tuple[FunctionSpec, ...], function_type: int | str | None) -> FunctionSpec:
        if function_type is None:
            return specs[0]
        for index, spec in enumerate(specs):
            if function_type == index or function_type == str(index) or function_type == spec.name:
                return spec
        return specs[0]


class ComplexApproximator(nn.Module):
    def __init__(
        self,
        first_day_of_week: int,
        patches_dates: Sequence[int] | None = None,
        main_function_type: int | str = 0,
        chain_functions_type: int | str = 0,
        connector_type: int | str = "mul",
        main_function_initial_weights: Sequence[float] | float | None = None,
        chain_functions_initial_weights: Sequence[Sequence[float] | float] | None = None,
        week_function_initial_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.main_function = ApproximatorsFactory.create_main_function(main_function_type, main_function_initial_weights)
        self.patches_dates = self._normalize_patch_dates(patches_dates)
        self.chain_functions = nn.ModuleList(
            [
                ApproximatorsFactory.create_chain_function(
                    chain_functions_type,
                    chain_functions_initial_weights[index] if chain_functions_initial_weights and index < len(chain_functions_initial_weights) else None,
                )
                for index in range(len(self.patches_dates))
            ]
        )
        self.connector, regularizer_base = ApproximatorsFactory.create_connector(connector_type)
        self.week_function = WeekFunction(first_day_of_week, regularizer_base, week_function_initial_weights)

    @staticmethod
    def _normalize_patch_dates(patches_dates: Sequence[int] | None) -> list[int]:
        if patches_dates is None:
            return []
        unique_dates = {int(patch_date) for patch_date in patches_dates if int(patch_date) > 0}
        return sorted(unique_dates)

    def init_weights_from_train_data(self, x_initial: Tensor, y_initial: Tensor) -> None:
        init_method = getattr(self.main_function, "init_weights_from_train_data")
        init_method(x_initial, y_initial)

    def forward(self, x: Tensor) -> Tensor:
        return self.connector(self.forward_trend_function(x), self.forward_week_function(x))

    def forward_trend_function(self, x: Tensor) -> Tensor:
        result = self.main_function(x)
        if not self.patches_dates:
            return result
        chain_mask = x >= self.patches_dates[-1]
        result[chain_mask] += self.chain_functions[-1](x[chain_mask])
        for index in range(len(self.chain_functions) - 2, -1, -1):
            chain_mask = (x >= self.patches_dates[index]) & (x < self.patches_dates[index + 1])
            result[chain_mask] += self.chain_functions[index](x[chain_mask])
        return result

    def forward_week_function(self, x: Tensor) -> Tensor:
        return self.week_function(x)

    def regularize(self) -> Tensor:
        return self.week_function.regularize()

    def summary_parameters(self) -> dict[str, list[float]]:
        summary: dict[str, list[float]] = {}
        for name, param in self.named_parameters():
            summary[name] = [float(value) for value in param.detach().cpu().flatten()]
        return summary
