import math
from collections.abc import Iterable, Mapping
from numbers import Real


_EACH = "*"
MetricPath = tuple[str, ...]


# Only these paths are evidence that an arm ran. Everything beside them is metadata
# owned by the runner that produced the payload and is deliberately outside this gate.
REQUIRED_ARM_METRIC_PATHS: dict[int, dict[str, tuple[MetricPath, ...]]] = {
    6: {
        "G1": (("measurements", _EACH, "by_model", _EACH, "one_step_test"),),
        "G2": (("measurements", _EACH, "by_model", _EACH, "one_step_test"),),
        "G3": (("measurements", _EACH, "by_model", _EACH, "one_step_test"),),
        "G4": (("measurements", _EACH, "by_model", _EACH, "one_step_test"),),
        "G5a": (("curves", _EACH, "principal"),),
        "G5b": (
            (
                "alpha_derivative",
                "by_model",
                _EACH,
                _EACH,
                "max_relative_error",
            ),
        ),
        "G6a": (("by_model", _EACH, "measurements", _EACH, "one_step_test"),),
        "G6b": (("by_model", _EACH, "by_dt", _EACH, "principal"),),
        "G7": (("varying_alpha", "band_ratio"), ("fixed_alpha", "band_ratio")),
        "G9": (("reference", "fraction_above_cutoff"),),
    },
    8: {
        "resolution_band_limited": (
            ("by_model", _EACH, "one_step_error"),
            ("by_model", _EACH, "rollout_error"),
        ),
        "resolution_new_high_k": (
            ("by_model", _EACH, "one_step_error"),
            ("by_model", _EACH, "rollout_error"),
        ),
        "robustness": (("model_errors",),),
    },
    9: {
        arm: tuple(
            ("measurements", _EACH, "relative_to_A", model)
            for model in ("B-loop", "C1", "C2", "C3")
        )
        for arm in ("sigma", "gamma")
    },
}


def _is_complete_finite_numeric_subtree(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return bool(value) and all(
            _is_complete_finite_numeric_subtree(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_complete_finite_numeric_subtree(item) for item in value)
    return False


def _required_values(root: object, path: MetricPath, *, context: str) -> list[object]:
    values = [(root, context)]
    declared_path = ".".join(path)
    for segment in path:
        resolved: list[tuple[object, str]] = []
        for value, location in values:
            if not isinstance(value, Mapping):
                raise RuntimeError(
                    f"{location} is not a mapping while resolving required finite "
                    f"metric path {declared_path}"
                )
            if segment == _EACH:
                if not value:
                    raise RuntimeError(
                        f"{location} is empty at required finite metric path "
                        f"{declared_path}"
                    )
                resolved.extend(
                    (item, f"{location}.{key}") for key, item in value.items()
                )
            else:
                if segment not in value:
                    raise RuntimeError(
                        f"{location}.{segment} is missing from required finite metric "
                        f"path {declared_path}"
                    )
                resolved.append((value[segment], f"{location}.{segment}"))
        values = resolved
    return [value for value, _ in values]


def require_phase_arms(phase: int, selected: Iterable[str], experiments: dict) -> None:
    if not isinstance(experiments, Mapping):
        raise RuntimeError(f"Phase {phase} experiments must be a mapping")
    if phase not in REQUIRED_ARM_METRIC_PATHS:
        raise RuntimeError(f"Phase {phase} has no payload metric schema")

    requirements = REQUIRED_ARM_METRIC_PATHS[phase]
    for arm in selected:
        if arm not in requirements:
            raise RuntimeError(f"Phase {phase} arm {arm} has no payload metric schema")
        if arm not in experiments or not experiments[arm]:
            raise RuntimeError(f"Phase {phase} arm {arm} has no measured result")

        context = f"Phase {phase} arm {arm}"
        for path in requirements[arm]:
            for value in _required_values(experiments[arm], path, context=context):
                if not _is_complete_finite_numeric_subtree(value):
                    raise RuntimeError(
                        f"{context}.{'.'.join(path)} needs a nonempty finite numeric "
                        "measurement"
                    )
