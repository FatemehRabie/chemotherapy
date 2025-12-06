"""Baseline control policies for the reaction-diffusion environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

import numpy as np


class BaselineController:
    """Interface for simple baseline controllers.

    Controllers are expected to expose a :meth:`predict` method compatible with
    Stable-Baselines3 models (i.e., returning ``(action, state)``). The default
    ``state`` is ``None`` so that controllers can be dropped into the existing
    evaluation loop.
    """

    name: str = "baseline"

    def reset(self, cell_line: int, env: Any) -> None:  # pragma: no cover - tiny helper
        """Hook to (re)initialise controller state before an episode."""

    def predict(
        self, observation: Any, deterministic: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError


@dataclass
class HeuristicScheduleController(BaselineController):
    """Simple rule-based dosing schedule.

    The controller increases the dose and duration when the tumour burden is
    high and tapers treatment as the tumour shrinks. It always uses a fixed
    drug (index ``drug_index``).
    """

    drug_index: int
    high_threshold: float = 0.75
    low_threshold: float = 0.25
    name: str = field(default="heuristic_schedule", init=False)

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        tumor_load = float(np.mean(observation["variables"][1]))
        if tumor_load > self.high_threshold:
            duration = 7
            dose = 9
        elif tumor_load > self.low_threshold:
            duration = 4
            dose = 6
        else:
            duration = 1
            dose = 2
        action = np.array([duration, dose, self.drug_index], dtype=int)
        return action, None


@dataclass
class PIDLikeTumorController(BaselineController):
    """PID-like controller that modulates dose based on tumour burden."""

    target: float = 0.2
    kp: float = 8.0
    ki: float = 1.0
    kd: float = 2.0
    drug_index: int = 0
    max_duration: int = 9
    name: str = field(default="pid_tumor", init=False)
    _integral: float = field(default=0.0, init=False)
    _previous_error: float = field(default=0.0, init=False)

    def reset(self, cell_line: int, env: Any) -> None:  # pragma: no cover - small state reset
        self._integral = 0.0
        self._previous_error = 0.0

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        tumor_load = float(np.mean(observation["variables"][1]))
        error = self.target - tumor_load
        self._integral += error
        derivative = error - self._previous_error
        control_signal = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._previous_error = error

        dose_continuous = np.clip(control_signal + 5.0, 0.0, 10.0)
        dose = int(round(dose_continuous))
        duration = int(round(np.clip(abs(control_signal), 0.0, self.max_duration)))
        action = np.array([duration, dose, self.drug_index], dtype=int)
        return action, None


@dataclass
class DrugRotationController(BaselineController):
    """Rotate through available drugs with a fixed moderate dose."""

    num_drugs: int
    rotation_interval: int = 2
    dose: int = 5
    duration: int = 3
    name: str = field(default="drug_rotation", init=False)
    _step_counter: int = field(default=0, init=False)

    def reset(self, cell_line: int, env: Any) -> None:  # pragma: no cover - trivial reset
        self._step_counter = 0

    def predict(self, observation: Any, deterministic: bool = True) -> Tuple[np.ndarray, None]:
        drug_index = (self._step_counter // self.rotation_interval) % self.num_drugs
        self._step_counter += 1
        action = np.array([self.duration, self.dose, drug_index], dtype=int)
        return action, None
