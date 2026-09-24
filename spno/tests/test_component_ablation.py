from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spno.checkpoints import CheckpointMetadata, save_checkpoint
from spno.config import DataConfig, config_hash
from spno.evaluation.component_ablation import (
    GAUGED_MODEL_NAMES, ComponentSplitStep, DensityPhaseSplitStep, component_models, crossed_interval,
    delta_q_profile, local_law, predicted_phase_drift,
    kinetic_dispersion, measure_rollout, probe_cases, reference_frames, sample_probe, state_metrics,
)
from spno.solvers.split_step import SplitStepNLSOperator
from scripts.run_hybrid_ablation import (DEFAULTS, load_c1_cohorts, multi_dt_data_hash,
                                        prolong, read_unit, run_study)


@pytest.fixture
def hybrid_source(tmp_path):
    cfg = replace(DataConfig(), grid_size=8, initial_bandwidth=2, steps=2, substeps=2,
                  n_train=2, n_val=2, n_test=2)
    source = tmp_path / "source"
    for cohort in ("base", "G6a", "G7-alpha-fixed"):
        identifier = config_hash(cfg)
        if cohort == "G6a":
            identifier = multi_dt_data_hash(cfg, quick=False)
        elif cohort == "G7-alpha-fixed":
            identifier = config_hash(replace(cfg, alpha_range=(.9, .9), seed=120))
        for seed in (0, 1):
            name = "C1" if cohort == "base" else f"{cohort}/C1"
            torch.manual_seed(seed)
            model = DensityPhaseSplitStep(cfg.domain, kinetic_mode="K0", width=4, trained_dt=cfg.dt)
            metadata = CheckpointMetadata(1, name, identifier, seed, "one-step", None, cfg.dt,
                                          {"kinetic_mode": "K0", "local_mode": "L0", "width": 4}, False, 0)
            save_checkpoint(source / "checkpoints/phase6" / (config_hash(cfg) + "-K0") / f"{name}-seed{seed}.pt", model, metadata)
    return source, cfg


def small_options():
    return {"training_seeds": [0, 1], "probe_seeds": [1000, 1001], "batch": 2,
            "bandwidths": [2, 4], "short_steps": 2, "long_steps": 4, "stride": 1,
            "reference_substeps": 2, "spatial_samples": 1, "bootstrap_draws": 20,
            "smoke": True}


def test_component_interventions_match_manual_composition_and_preserve_source():
    cfg = replace(DataConfig(), grid_size=16, initial_bandwidth=4)
    model = DensityPhaseSplitStep(cfg.domain, width=4).double()
    original = {k: v.clone() for k, v in model.state_dict().items()}
    case = probe_cases(cfg, [4])[0]
    inputs, _ = sample_probe(case, 1000, 3)
    x, potential, alpha, beta = inputs
    parameters = torch.stack([alpha, beta], -1)
    solver = SplitStepNLSOperator(cfg.domain)
    models = component_models(model)
    assert torch.allclose(models["exact_split"](*inputs, cfg.dt), solver(*inputs, cfg.dt), atol=1e-13, rtol=1e-13)
    for name, exact_k, exact_l in (("C1", False, False), ("exactK_learnedL", True, False), ("learnedK_exactL", False, True)):
        rate = -alpha[:, None] * cfg.domain.wave_number_squared() if exact_k else model.kinetic(parameters)
        half = lambda state: torch.fft.ifft(torch.fft.fft(state) * torch.exp(.5j * cfg.dt * rate))
        mid = half(x)
        local = beta[:, None]*mid.abs().square() - potential if exact_l else model.local_phase(mid, potential, alpha, beta)
        expected = half(mid * torch.exp(1j*cfg.dt*local))
        actual = models[name](*inputs, cfg.dt)
        assert torch.allclose(actual, expected, atol=1e-13, rtol=1e-13)
        assert torch.allclose(models[name](actual, potential, alpha, beta, -cfg.dt), x, atol=1e-12, rtol=1e-12)
    assert all(torch.equal(v, original[k]) for k, v in model.state_dict().items())
    assert models["exactK_learnedL"].local is not model.local


def test_phase_metrics_distinguish_global_phase_from_density_error():
    cfg = replace(DataConfig(), grid_size=8, initial_bandwidth=2)
    inputs, _ = sample_probe(probe_cases(cfg, [2])[0], 1000, 2)
    x, potential, alpha, beta = inputs
    angle = .7
    pred = x * np.exp(1j*angle)
    metrics = state_metrics(pred, x, x, potential, alpha, beta, cfg.domain)
    assert torch.allclose(metrics["state_error"], torch.full((2,), 2*np.sin(angle/2), dtype=torch.float64))
    assert metrics["aligned_state_error"].max() < 1e-14
    assert torch.allclose(metrics["phase_rms"], torch.full((2,), angle, dtype=torch.float64))
    assert metrics["spectrum_error"].max() < 1e-14
    assert metrics["energy_error"].max() < 1e-14


def test_probe_seeds_change_ics_but_bandwidth_sweep_preserves_parameters_and_mass():
    cfg = replace(DataConfig(), grid_size=16, initial_bandwidth=4)
    cases = {c.name: c for c in probe_cases(cfg, [4, 6])}
    (x, v, a, b), h = sample_probe(cases["G4-bandwidth-4"], 1000, 3)
    same, same_hash = sample_probe(cases["G4-bandwidth-4"], 1000, 3)
    changed, different_hash = sample_probe(cases["G4-bandwidth-4"], 1001, 3)
    wider, _ = sample_probe(cases["G4-bandwidth-6"], 1000, 3)
    assert h == same_hash and h != different_hash
    assert torch.equal(x, same[0]) and not torch.equal(x, changed[0])
    for actual, expected in zip(wider[1:], (v, a, b)):
        assert torch.equal(actual, expected)
    assert torch.allclose(wider[0].abs().square().sum(-1), x.abs().square().sum(-1))
    k = torch.fft.fftfreq(16, d=1/16)
    assert torch.fft.fft(x)[..., k.abs() > 4].abs().max() < 1e-13


def test_generator_dispersion_has_no_phase_wrapping_and_correct_sign():
    cfg = DataConfig()
    model = DensityPhaseSplitStep(cfg.domain, kinetic_mode="K2", local_mode="L2").double()
    payload = kinetic_dispersion(model, cfg)
    assert len(payload["k"]) == cfg.grid_size
    for curve in payload["curves"]:
        assert np.allclose(curve["omega"], curve["exact"], atol=1e-12)
        assert np.max(np.abs(curve["alpha_secant_error"])) < 1e-10
        assert np.max(curve["map_residual"]["exactK_learnedL"]) < 1e-12
    assert np.max(payload["curves"][0]["omega"]) > np.pi/cfg.dt


def test_prolongation_recovers_samples_and_records_failures():
    cfg = replace(DataConfig(), grid_size=8, initial_bandwidth=2)
    inputs, _ = sample_probe(probe_cases(cfg, [2])[0], 1000, 2)
    assert torch.allclose(prolong(inputs[0])[..., ::2], inputs[0], atol=1e-14)
    truth = reference_frames(inputs, cfg, steps=3, stride=1, substeps=2)
    class Fails:
        def __call__(self, state, *args):
            result = state.clone()
            result[0] = torch.nan
            return result
    result = measure_rollout(Fails(), inputs, cfg, truth, training_bandwidth=2)
    assert result["failed_at"] == [1, None]
    assert np.isnan(result["records"][-1]["state_error"][0])
    assert np.isfinite(result["records"][-1]["state_error"][1])


def test_crossed_interval_retains_seed_uncertainty():
    # IC duplication cannot eliminate training-seed variability.
    small = np.broadcast_to(np.array([-1., 1.])[:, None, None], (2, 3, 1))
    many = np.repeat(small, 100, axis=-1)
    for value in (small, many):
        result = crossed_interval(value, draws=500)
        assert result["low"] < -.5 and result["high"] > .5
    assert crossed_interval(np.full((2, 3, 2), .5), draws=20)["mean"] == .5
    assert crossed_interval(np.full((2, 3, 2), np.nan))["mean"] is None


def test_full_suite_saves_paired_results_and_resumes_without_reference_work(hybrid_source, tmp_path, monkeypatch):
    source, cfg = hybrid_source
    before = {p: p.read_bytes() for p in source.rglob("*.pt")}
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("Implicit training"))
    run = run_study(source, tmp_path / "output", data=cfg, options=small_options())
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["complete"] and manifest["exploratory"]
    assert {c["name"].split("-")[0] for c in manifest["cases"]} >= {"G1", "G2", "G3", "G4", "G6a", "G6b", "G7", "G8", "G9"}
    assert read_unit(run / "dispersion.json.gz")["base"]
    files = list((run / "cases").rglob("base-0.json.gz"))
    assert files
    for path in files:
        first = read_unit(path)
        second = read_unit(path.with_name("base-1.json.gz"))
        assert first["input_sha256"] == second["input_sha256"]
        assert first["by_model"]["exact_split"] == second["by_model"]["exact_split"]
    monkeypatch.setattr("scripts.run_hybrid_ablation.reference_bundle", lambda *a, **k: pytest.fail("Recomputed completed unit"))
    again = run_study(source, tmp_path / "output", data=cfg, options=small_options())
    assert again == run
    assert all(p.read_bytes() == value for p, value in before.items())


def test_notebook_executes_every_cell(hybrid_source, tmp_path, monkeypatch):
    nbformat = pytest.importorskip("nbformat")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    from dataclasses import asdict
    source, cfg = hybrid_source
    project = Path(__file__).resolve().parents[1]
    config_path = tmp_path / "source.json"
    config_path.write_text(json.dumps({"data": asdict(cfg)}))
    settings = small_options()
    # Exercise every arm in the runner test above; render representative figures here.
    settings["case_names"] = ["G1-interpolation", "G3-potential-well", "G4-bandwidth-2",
                              "G6a-multidt-1", "G7-alpha-fixed", "G8-long-rollout-extension", "G9-cascade-long"]
    for key, value in {"SPNO_PROJECT_ROOT": project, "SPNO_SOURCE_ROOT": source,
                       "SPNO_SOURCE_CONFIG": config_path, "SPNO_OUTPUT_ROOT": tmp_path / "notebook-output",
                       "SPNO_OPTIONS": json.dumps(settings), "MPLBACKEND": "Agg",
                       "IPYTHONDIR": tmp_path / "ipython", "JUPYTER_RUNTIME_DIR": tmp_path / "jupyter"}.items():
        monkeypatch.setenv(key, str(value))
    notebook = nbformat.read(project / "notebooks/11_hybrid_kinetic_ablation.ipynb", as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import torch\ndef forbidden(*a, **k): raise AssertionError('Implicit training')\ntorch.optim.AdamW.step=forbidden"))
    executed = NotebookClient(notebook, timeout=300, kernel_name="python3",
                              resources={"metadata": {"path": str(project)}}).execute()
    assert not any(o.output_type == "error" for c in executed.cells if c.cell_type == "code" for o in c.outputs)
    assert list((tmp_path / "notebook-output").rglob("01_all_arms_hybrid_vs_C1.pdf"))
    assert list((tmp_path / "notebook-output").rglob("paired_comparisons.csv"))


def _random_c1(cfg, **kwargs):
    torch.manual_seed(3)
    model = DensityPhaseSplitStep(cfg.domain, width=4, **kwargs).double()
    # Break the zero init of any correction head so no rung is trivially exact.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.3 * torch.randn_like(parameter))
    return model.eval()


def test_gauge_move_is_an_exact_identity_on_c1():
    cfg = replace(DataConfig(), grid_size=16, initial_bandwidth=4)
    model = _random_c1(cfg)
    inputs, _ = sample_probe(probe_cases(cfg, [4])[0], 1000, 3)
    both = ComponentSplitStep(model, exact_kinetic=False, exact_local=False, gauge="zero_mode")
    torch.testing.assert_close(both(*inputs, cfg.dt), model(*inputs, cfg.dt), atol=1e-12, rtol=1e-12)
    parameters = torch.stack(inputs[2:], -1)
    assert model.kinetic(parameters)[:, 0].abs().min() > 0  # the gauge move is not vacuous


def test_gauged_swaps_move_the_offset_and_keep_structure():
    cfg = replace(DataConfig(), grid_size=16, initial_bandwidth=4)
    model = _random_c1(cfg)
    x, potential, alpha, beta = inputs = sample_probe(probe_cases(cfg, [4])[0], 1000, 3)[0]
    models = component_models(model, gauge=True)
    assert set(GAUGED_MODEL_NAMES) <= set(models)
    parameters = torch.stack((alpha, beta), -1)
    offset = model.kinetic(parameters)[:, :1]
    mid = models["exact_split"]._kinetic_half(x, parameters, cfg.dt)
    assert torch.allclose(models["exactK_learnedL_g"].local_phase(mid, potential, alpha, beta),
                          model.local_phase(mid, potential, alpha, beta) + offset, atol=1e-13)
    # The reverse swap drops the k=0 rate, so the zero mode is untouched by the kinetic half.
    constant = torch.ones_like(x)
    assert torch.allclose(models["learnedK_exactL_g"]._kinetic_half(constant, parameters, cfg.dt),
                          constant, atol=1e-13)
    for name in GAUGED_MODEL_NAMES:
        out = models[name](*inputs, cfg.dt)
        assert torch.allclose(out.abs().square().sum(-1), x.abs().square().sum(-1), rtol=1e-13)
        rotated = models[name](x * np.exp(.4j), potential, alpha, beta, cfg.dt)
        assert torch.allclose(rotated, out * np.exp(.4j), atol=1e-13)


def test_local_law_recovers_the_exact_rate():
    cfg = replace(DataConfig(), grid_size=16, initial_bandwidth=4)
    exact = DensityPhaseSplitStep(cfg.domain, kinetic_mode="K2", local_mode="L2", width=4).double()
    inputs, _ = sample_probe(probe_cases(cfg, [4])[0], 1000, 8)
    law = local_law(exact, inputs)
    for key, target in zip(("beta_rho", "potential", "intercept", "alpha"), law["truth"]):
        assert abs(law[key] - target) < 1e-10
    assert law["relative_rmse_vs_truth"] < 1e-12
    profile = delta_q_profile(exact, inputs[0].abs().square().numpy())
    assert max(abs(f["slope"]) + abs(f["offset"]) for f in profile["fits"]) < 1e-10


def test_local_law_and_drift_predictor_detect_a_known_bias():
    cfg = replace(DataConfig(), grid_size=16, initial_bandwidth=4)
    model = DensityPhaseSplitStep(cfg.domain, kinetic_mode="K2", local_mode="L2", width=4).double()
    with torch.no_grad():
        model.local.network.network[-1].bias.fill_(.1)  # delta q = +0.1 everywhere
    inputs, _ = sample_probe(probe_cases(cfg, [4])[0], 1000, 4)
    assert abs(local_law(model, inputs)["intercept"] - .1) < 1e-10
    drift = predicted_phase_drift(model, inputs, cfg.dt, steps=20, reference_substeps=8)
    for stats in drift["stats"].values():
        assert abs(stats["slope"] - 1) < 1e-2 and stats["corr"] > .999


GAUGE_CASES = ["G1-interpolation", "G2-extrapolation", "G3-potential-strong", "G4-bandwidth-2", "G6a-multidt-1"]


def test_gauge_run_saves_every_swap_all_pairs_and_the_reciprocal_figure(hybrid_source, tmp_path):
    from scripts.plot_hybrid_ablation import export_plots
    source, cfg = hybrid_source
    options = {**small_options(), "gauge": True, "case_names": GAUGE_CASES}
    run = run_study(source, tmp_path / "output", data=cfg, options=options)
    manifest = json.loads((run / "manifest.json").read_text())
    assert set(GAUGED_MODEL_NAMES) <= set(manifest["models"])
    unit = read_unit(next((run / "cases").rglob("base-0.json.gz")))
    assert set(unit["by_model"]) == set(manifest["models"])
    summary = json.loads((run / "summary.json").read_text())
    pairs = {(r["model"], r["baseline"]) for r in summary["paired"]}
    assert ("exactK_learnedL", "learnedK_exactL") in pairs and ("learnedK_exactL_g", "C1") in pairs
    legacy = [r for r in summary["paired"] if "hybrid_over_C1" in r]
    assert legacy and all((r["model"], r["baseline"]) == ("exactK_learnedL", "C1") for r in legacy)
    figures = export_plots(run)
    assert run / "figures" / "07_reciprocal_ablation.png" in figures


def test_resummarize_keeps_run_identity_and_adds_reciprocal_pairs(hybrid_source, tmp_path):
    from scripts.run_hybrid_ablation import resummarize
    source, cfg = hybrid_source
    run = run_study(source, tmp_path / "output", data=cfg,
                    options={**small_options(), "case_names": GAUGE_CASES[:2]})
    manifest = json.loads((run / "manifest.json").read_text())
    del manifest["models"]  # a pre-gauge manifest, like Drive run 3bf81deae4ef1ff9
    (run / "manifest.json").write_text(json.dumps(manifest))
    old = json.loads((run / "summary.json").read_text())
    summary = resummarize(run)
    assert json.loads((run / "summary.v1.json").read_text()) == old
    assert {(r["model"], r["baseline"]) for r in summary["paired"]} == {
        ("exactK_learnedL", "C1"), ("learnedK_exactL", "C1"), ("exactK_learnedL", "learnedK_exactL")}
    assert sorted(run.parent.iterdir()) == [run]
    assert summary["rows"] == old["rows"]


def test_substituted_c1g_cohort_runs_base_rows_and_gauged_swaps_are_identities(hybrid_source, tmp_path):
    source, cfg = hybrid_source
    models = {}
    for seed in (0, 1):
        torch.manual_seed(seed)
        models[seed] = DensityPhaseSplitStep(cfg.domain, width=4, kinetic_gauge="zero_mode").double().eval()
    provenance = [{"name": "C1g", "seed": s, "sha256": "test", "metadata": {"converged": False}, "quick": True}
                  for s in (0, 1)]
    options = {**small_options(), "gauge": True, "case_names": GAUGE_CASES}
    run = run_study(source, tmp_path / "output", data=cfg, options=options,
                    cohorts=({"base": models}, provenance))
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["checkpoint_family"] == ["C1g"]
    assert all(c["cohorts"] == ["base"] for c in manifest["cases"])
    for path in (run / "cases").rglob("base-*.json.gz"):
        by_model = read_unit(path)["by_model"]
        # kappa(0) = 0 already, so the gauge move changes nothing.
        for name in GAUGED_MODEL_NAMES:
            ungauged = by_model[name[:-2]]["records"][-1]["state_error"]
            assert np.allclose(by_model[name]["records"][-1]["state_error"], ungauged, rtol=1e-10, atol=1e-14)


def test_load_c1g_cohort_reads_workflow_records(tmp_path):
    from spno.artifacts import file_digest
    from spno.workflow import stable_hash
    from scripts.run_hybrid_ablation import load_c1g_cohort
    cfg = replace(DataConfig(), grid_size=8, initial_bandwidth=2)
    architecture = {"kinetic_mode": "K0", "local_mode": "L0", "width": 4, "kinetic_gauge": "zero_mode"}
    for seed, weight in ((0, 0.), (0, .01), (1, 0.)):
        spec = {"name": "C1g", "seed": seed, "physics_weight": weight, "data_hash": config_hash(cfg), "quick": False}
        folder = tmp_path / "experiments" / stable_hash(spec)
        model = DensityPhaseSplitStep(cfg.domain, width=4, kinetic_gauge="zero_mode")
        meta = CheckpointMetadata(1, "C1g", config_hash(cfg), seed, "one-step", None, cfg.dt, architecture, False, 0)
        save_checkpoint(folder / "model.pt", model, meta)
        (folder / "record.json").write_text(json.dumps({"spec": spec, "sha256": file_digest(folder / "model.pt")}))
    cohorts, provenance = load_c1g_cohort(tmp_path, cfg, [0, 1])
    assert set(cohorts["base"]) == {0, 1} and [p["seed"] for p in provenance] == [0, 1]
    assert all(m.kinetic.gauge == "zero_mode" for m in cohorts["base"].values())
    with pytest.raises(ValueError, match="missing for seeds \\[2\\]"):
        load_c1g_cohort(tmp_path, cfg, [0, 2])
