import math
from collections.abc import Iterable


REQUIRED_ARM_KEYS = {
    6: {
        "G1": ("measurements",),
        "G2": ("measurements",),
        "G3": ("measurements",),
        "G4": ("measurements",),
        "G5a": ("curves",),
        "G5b": ("alpha_derivative",),
        "G6a": ("by_model",),
        "G6b": ("by_model",),
        "G7": ("varying_alpha", "fixed_alpha"),
        "G9": ("reference",),
    },
    8: {
        "resolution_band_limited": ("by_model",),
        "resolution_new_high_k": ("by_model",),
        "robustness": ("model_errors",),
    },
    9: {"sigma": ("measurements",), "gamma": ("measurements",)},
}

PROVENANCE_KEYS = frozenset({"identifier", "potential_family", "note"})


def _has_complete_finite_measurements(value) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return bool(value) and all(
            _has_complete_finite_measurements(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_has_complete_finite_measurements(item) for item in value)
    return False


def _has_complete_phase_six_measurements(value) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    for record in value.values():
        if not isinstance(record, dict):
            return False
        metrics = {key: item for key, item in record.items() if key not in PROVENANCE_KEYS}
        if not _has_complete_finite_measurements(metrics):
            return False
    return True


def require_phase_arms(phase: int, selected: Iterable[str], experiments: dict) -> None:
    requirements = REQUIRED_ARM_KEYS[phase]
    for arm in selected:
        if arm not in experiments or not experiments[arm]:
            raise RuntimeError(f"Phase {phase} arm {arm} has no measured result")
        for key in requirements[arm]:
            if key not in experiments[arm]:
                raise RuntimeError(
                    f"Phase {phase} arm {arm}.{key} needs a finite measurement"
                )
            if phase == 6 and arm in {"G1", "G2", "G3", "G4"} and key == "measurements":
                has_measurements = _has_complete_phase_six_measurements(
                    experiments[arm][key]
                )
            else:
                has_measurements = _has_complete_finite_measurements(experiments[arm][key])
            if not has_measurements:
                raise RuntimeError(
                    f"Phase {phase} arm {arm}.{key} needs a finite measurement"
                )
        if phase == 6 and arm in {"G1", "G2", "G3", "G4"}:
            for shift, result in experiments[arm]["measurements"].items():
                if not result.get("by_model") or not _has_complete_finite_measurements(
                    result["by_model"]
                ):
                    raise RuntimeError(
                        f"Phase 6 arm {arm}/{shift} needs finite model metrics"
                    )
        if phase == 9:
            expected_models = {"B-loop", "C1", "C2", "C3"}
            for value, result in experiments[arm]["measurements"].items():
                ratios = result.get("relative_to_A", {})
                if set(ratios) != expected_models or not all(
                    isinstance(ratio, (int, float)) and math.isfinite(float(ratio))
                    for ratio in ratios.values()
                ):
                    raise RuntimeError(
                        f"Phase 9 arm {arm}/{value} needs complete finite A-relative ratios"
                    )
