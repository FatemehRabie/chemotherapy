import numpy as np


class BasePolicy:
    """Simple interface to align heuristic policies with Stable Baselines models."""

    name = "base"

    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=True):  # pylint: disable=unused-argument
        raise NotImplementedError


class FixedSchedulePolicy(BasePolicy):
    name = "FixedSchedule"

    def __init__(self, action_space, duration=1, dose=5, drug_type=0):
        super().__init__(action_space)
        max_duration, max_dose, max_drug = action_space.nvec
        self.duration = int(np.clip(duration, 0, max_duration - 1))
        self.dose = int(np.clip(dose, 0, max_dose - 1))
        self.drug_type = int(np.clip(drug_type, 0, max_drug - 1))

    def predict(self, observation, deterministic=True):  # pylint: disable=unused-argument
        action = np.array([self.duration, self.dose, self.drug_type], dtype=np.int64)
        return action, None


class TumorProportionalPolicy(BasePolicy):
    name = "ProportionalControl"

    def __init__(self, action_space, duration_scale=4.0, dose_scale=8.0, drug_type=0):
        super().__init__(action_space)
        self.duration_scale = duration_scale
        self.dose_scale = dose_scale
        self.drug_type = drug_type
        self.max_duration, self.max_dose, self.max_drug = action_space.nvec

    def predict(self, observation, deterministic=True):  # pylint: disable=unused-argument
        tumor_load = float(np.mean(observation["variables"][1]))
        duration = int(np.clip(np.round(tumor_load * self.duration_scale), 0, self.max_duration - 1))
        dose = int(np.clip(np.round(tumor_load * self.dose_scale), 0, self.max_dose - 1))
        drug = int(np.clip(self.drug_type, 0, self.max_drug - 1))
        action = np.array([duration, dose, drug], dtype=np.int64)
        return action, None


class RandomPolicy(BasePolicy):
    name = "RandomPolicy"

    def __init__(self, action_space, seed=None):
        super().__init__(action_space)
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, deterministic=True):  # pylint: disable=unused-argument
        if hasattr(self.action_space, "nvec"):
            action = self.rng.integers(low=0, high=self.action_space.nvec)
        else:
            action = self.action_space.sample()
        return action, None
