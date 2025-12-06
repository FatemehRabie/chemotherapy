"""Simple baseline policies for chemotherapy control."""

from baselines.policies import (
    FixedSchedulePolicy,
    RandomPolicy,
    TumorProportionalPolicy,
)

BASELINE_FACTORIES = {
    "FIXEDSCHEDULE": lambda action_space, **kwargs: FixedSchedulePolicy(action_space, **kwargs),
    "PROPORTIONALCONTROL": lambda action_space, **kwargs: TumorProportionalPolicy(action_space, **kwargs),
    "RANDOMPOLICY": lambda action_space, **kwargs: RandomPolicy(action_space, seed=kwargs.get("seed")),
}


def is_baseline(name: str) -> bool:
    return name.upper() in BASELINE_FACTORIES


def make_baseline_policy(name: str, action_space, **kwargs):
    key = name.upper()
    if key not in BASELINE_FACTORIES:
        raise ValueError(f"Unknown baseline policy '{name}'")
    return BASELINE_FACTORIES[key](action_space, **kwargs)


__all__ = [
    "FixedSchedulePolicy",
    "RandomPolicy",
    "TumorProportionalPolicy",
    "is_baseline",
    "make_baseline_policy",
]
