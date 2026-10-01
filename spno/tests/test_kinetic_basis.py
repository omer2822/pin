from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from spno.config import DataConfig
from spno.models.split_learned import DensityPhaseSplitStep
from test_workflow import standalone


def test_low_band_fit_extrapolates_a_learned_law_and_preserves_gauge():
    from scripts.run_kinetic_basis import fit_polynomial
    cfg = DataConfig()
    source = DensityPhaseSplitStep(cfg.domain).double()

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.domain = cfg.domain
            self.register_buffer('k_squared', source.kinetic.k_squared.clone())

        def forward(self, p):
            # Deliberately differs from the physical law: fit must recover the teacher,
            # not silently install -alpha*k^2.
            coefficient = .8 - .5 * p[:, 0] + .2 * p[:, 1]
            offset = 3 + p[:, 1].square()
            return offset[:, None] + coefficient[:, None] * self.k_squared

    source.kinetic = Teacher()
    before = {k: v.clone() for k, v in source.state_dict().items()}
    parameters = torch.tensor([[.73, -.2], [.95, .17], [1.05, .43]], dtype=torch.float64)
    for degree in (1, 3):
        repaired, info = fit_polynomial(source, cfg, degree=degree, fit_k=8)
        assert info['fit_k'] == 8 and info['fit_rmse'] < 1e-9
        assert torch.allclose(repaired.kinetic(parameters), source.kinetic(parameters), atol=1e-7)
        assert torch.equal(repaired.kinetic(parameters)[:, 0], source.kinetic(parameters)[:, 0])
        assert all(torch.equal(v, before[k]) for k, v in source.state_dict().items())
        field = torch.randn(3, 64, dtype=torch.complex128)
        potential = torch.zeros(3, 64, dtype=torch.float64)
        updated = repaired(field, potential, parameters[:, 0], parameters[:, 1], cfg.dt)
        assert torch.allclose(updated.abs().square().sum(-1), field.abs().square().sum(-1), atol=1e-10)
        recovered = repaired(updated, potential, parameters[:, 0], parameters[:, 1], -cfg.dt)
        assert torch.allclose(recovered, field, atol=1e-10)


def test_notebook_executes_without_optimizer_steps(standalone, tmp_path, monkeypatch):
    nbformat = pytest.importorskip('nbformat')
    NotebookClient = pytest.importorskip('nbclient').NotebookClient
    source, data, _, _ = standalone
    project = Path(__file__).resolve().parents[1]
    config = tmp_path / 'source.json'
    config.write_text(json.dumps({'data': asdict(data)}))
    options = {'seeds': [3], 'probe_seeds': [1000], 'probe_batch': 2,
               'rollout_steps': 2, 'stride': 1, 'reference_substeps': 2,
               'bandwidths': [2, 3], 'fit_k': 2, 'degrees': [1], 'smoke': True}
    for key, value in {'SPNO_PROJECT_ROOT': project, 'SPNO_SOURCE_ROOT': source,
                       'SPNO_SOURCE_CONFIG': config, 'SPNO_OUTPUT_ROOT': tmp_path / 'output',
                       'SPNO_OPTIONS': json.dumps(options), 'MPLBACKEND': 'Agg',
                       'IPYTHONDIR': tmp_path / 'ipython',
                       'JUPYTER_RUNTIME_DIR': tmp_path / 'jupyter'}.items():
        monkeypatch.setenv(key, str(value))
    notebook = nbformat.read(project / 'notebooks/14_kinetic_basis_quick_check.ipynb', as_version=4)
    nbformat.validate(notebook)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        'import torch\ndef forbidden(*a, **kw):\n'
        '    raise AssertionError("default must not use gradient training")\n'
        'torch.optim.AdamW.step = forbidden'))
    executed = NotebookClient(notebook, timeout=180, kernel_name='python3',
                              resources={'metadata': {'path': str(project)}}).execute()
    assert not [o for c in executed.cells if c.cell_type == 'code'
                for o in c.outputs if o.output_type == 'error']
    summary_path = next((tmp_path / 'output').rglob('summary.json'))
    summary = json.loads(summary_path.read_text())
    assert summary['exploratory'] and summary['options']['head_epochs'] == 0
    assert {r['arm'] for r in summary['rows']} == {'K0-tanh', 'poly1'}
    assert all(r['verdict'] == 'SMOKE ONLY' for r in summary['rows'])
    assert list(summary_path.parent.glob('*.png'))


def test_optional_training_changes_only_coefficient_head(standalone):
    from scripts.run_kinetic_basis import DEFAULTS, fit_polynomial, fine_tune_head
    source_root, cfg, _, originals = standalone
    source = originals['C1', 3]
    model, _ = fit_polynomial(source, cfg, degree=1, fit_k=2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    tuned, history = fine_tune_head(model, source_root, cfg,
                                   {**DEFAULTS, 'head_epochs': 2, 'head_pairs': 4}, 3)
    assert len(history['train_loss']) == 2
    changed = {k for k, v in tuned.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and changed <= {'kinetic.coefficients.weight', 'kinetic.coefficients.bias'}
