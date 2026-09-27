from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from spno.checkpoints import load_checkpoint_payload
from spno.config import DataConfig
import scripts.run_kinetic_support as ks
from scripts.run_phase6 import _model_from_checkpoint


TINY = replace(DataConfig(), grid_size=8, initial_bandwidth=2, steps=3, substeps=2,
               n_train=2, n_val=2, n_test=2)
CELLS = [{"kinetic": "K0", "bandwidth": 2}, {"kinetic": "K0", "bandwidth": 3},
         {"kinetic": "K1", "bandwidth": 2}, {"kinetic": "K2", "bandwidth": 2}]
OPTIONS = {"cells": CELLS, "seeds": [0], "epochs": 2, "device": "cpu", "probe_k": [2, 4],
           "rollout_bandwidths": [2, 3], "rollout_steps": 2, "check_steps": 2,
           "check_trajectories": 1, "probe_batch": 2, "reference_substeps": 2, "smoke": True}


def test_knee_is_the_first_nonzero_mode_above_threshold():
    residual = [0.9, 1e-4, 2e-3, 0.03, 0.2, 0.6]
    assert ks.knee(residual, 0.01) == 3
    assert ks.knee(residual, 0.05) == 4
    assert ks.knee(residual, 1.0) is None  # k = 0 never counts, and nothing exceeds 1
    assert ks.as_knee(None, 64) == 33 and ks.as_knee(7, 64) == 7


def test_k2_starts_on_the_exact_rate_and_every_rung_round_trips(tmp_path):
    study = ks.prepare(tmp_path / "out", tmp_path / "data", base=TINY, options=OPTIONS)
    for cell in study["cells"]:
        model = ks.build_model(cell, seed=0)
        if cell["kinetic"] == "K2":
            k, _, _, centered = ks.plane_wave_rates(model, cell["config"], .9, .3)
            assert torch.allclose(centered, -.9 * k.square(), atol=1e-12)
            assert ks.output_bound(model) is None
        else:
            assert ks.output_bound(model) > 0
    ks.run_data(study)
    ks.train_all(study)
    for cell in study["cells"]:
        path = study["root"] / "checkpoints" / cell["name"] / "C1-seed0.pt"
        payload = load_checkpoint_payload(path)
        assert payload.metadata.architecture["kinetic_mode"] == cell["kinetic"]
        rebuilt = _model_from_checkpoint(payload, cell["config"], expected_name="C1")
        rebuilt.load_state_dict(payload.state_dict, strict=True)
        loaded, _ = ks.load_trained(study, cell, 0)
        assert rebuilt.kinetic.mode == loaded.kinetic.mode == cell["kinetic"]


def test_same_seed_reproduces_the_phase6_c1_initialization():
    cell = {"kinetic": "K0", "bandwidth": 8, "name": "K0-bw8", "config": DataConfig()}
    torch.manual_seed(1)
    from spno.models.split_learned import DensityPhaseSplitStep
    phase6 = DensityPhaseSplitStep(DataConfig().domain, kinetic_mode="K0", local_mode="L0",
                                   width=32, trained_dt=.01)
    ours = ks.build_model(cell, seed=1)
    assert all(torch.equal(a, b) for a, b in zip(phase6.state_dict().values(), ours.state_dict().values()))


def test_full_run_resumes_without_retraining_and_writes_every_output(tmp_path, monkeypatch):
    study = ks.prepare(tmp_path / "out", tmp_path / "data", base=TINY, options=OPTIONS)
    checks = ks.run_data(study)
    assert all(entry["support"]["1e-06"] >= entry["bandwidth"] for entry in checks.values())
    ks.train_all(study)

    def forbidden(*args, **kwargs):
        raise AssertionError("retrained a finished run")
    monkeypatch.setattr(ks, "train_one_step", forbidden)
    again = ks.prepare(tmp_path / "out", tmp_path / "data", base=TINY, options=OPTIONS)
    assert again["run_id"] == study["run_id"]
    ks.run_data(again)
    ks.train_all(again)
    port = ks.measure_all(again)
    assert port["status"] == "skipped"
    summary = ks.summarize(again)
    assert set(summary["cells"]) == {ks.cell_name(c) for c in CELLS}
    assert summary["verdict"]["data_support"] == "not evaluable"
    k2 = summary["cells"]["K2-bw2"]
    assert k2["plateau"][0] == pytest.approx(-.9 * 16, rel=.05)  # K2 barely moves from -alpha k^2
    assert (again["root"] / "knees.csv").read_text().startswith("cell,")
    from scripts.plot_kinetic_support import export_plots
    figures = export_plots(again["root"])
    assert len(figures) == 5 and all(f.exists() for f in figures)


def _cells(knees, gains):
    return {name: {"kinetic": "K0", "knees": {"0.9/0.05": values}, "data_resolved": True,
                   "budget": [{"last5_improvement": gains.get(name, 0.0)}] * len(values)}
            for name, values in knees.items()}


@pytest.mark.parametrize("knees, gains, expected", [
    ({"K0-bw8": [10, 10, 10], "K0-bw10": [12, 11, 12], "K0-bw12": [14, 13, 11]}, {}, "data support SUPPORTED"),
    ({"K0-bw8": [10, 10, 10], "K0-bw10": [10, 11, 10], "K0-bw12": [10, 11, 9]}, {}, "architecture limit INDICATED"),
    ({"K0-bw8": [10, 10, 10], "K0-bw12": [10, 11, 9]}, {"K0-bw12": .2}, "INCONCLUSIVE: knees did not move"),
    ({"K0-bw8": [10, 10, 10], "K0-bw12": [12, 10, 10]}, {}, "INCONCLUSIVE"),
])
def test_preregistered_verdict(knees, gains, expected):
    cells = [{"kinetic": "K0", "bandwidth": int(n.split("bw")[1]), "name": n} for n in knees]
    study = {"base": DataConfig(), "cells": cells}
    assert ks.verdict(study, _cells(knees, gains))["data_support"].startswith(expected)


def test_notebook_executes_every_cell(tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    project = Path(__file__).resolve().parents[1]
    base = {k: v for k, v in asdict(TINY).items() if k in ("grid_size", "initial_bandwidth", "steps",
                                                               "substeps", "n_train", "n_val", "n_test")}
    for key, value in {"SPNO_PROJECT_ROOT": project, "SPNO_OUTPUT_ROOT": tmp_path / "output",
                       "SPNO_DATA_ROOT": tmp_path / "data", "SPNO_OPTIONS": json.dumps(OPTIONS),
                       "SPNO_BASE": json.dumps(base), "MPLBACKEND": "Agg",
                       "IPYTHONDIR": tmp_path / "ipython", "JUPYTER_RUNTIME_DIR": tmp_path / "jupyter"}.items():
        monkeypatch.setenv(key, str(value))
    notebook = nbformat.read(project / "notebooks/13_kinetic_support_ladder.ipynb", as_version=4)
    nbformat.validate(notebook)
    executed = NotebookClient(notebook, timeout=600, kernel_name="python3",
                              resources={"metadata": {"path": str(project)}}).execute()
    errors = [o for c in executed.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
    assert not errors, errors
    assert list((tmp_path / "output").rglob("02_knee_vs_bandwidth.pdf"))
    assert list((tmp_path / "output").rglob("summary.json"))
