from pathlib import Path
import json

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient

from test_workflow import budget_bound_standalone, standalone


@pytest.mark.parametrize('filename', ['00_training.ipynb', '07_pino_evaluation.ipynb',
                                      '08_resolution_robustness.ipynb', '09_misspecification.ipynb',
                                      '10_dirichlet.ipynb'])
def test_notebook_executes_end_to_end_without_training(filename, standalone, tmp_path, monkeypatch):
    from dataclasses import asdict
    root, data, config, _ = standalone
    project = Path(__file__).resolve().parents[1]
    path = project / 'notebooks' / filename
    assert path.is_file(), f'Missing deliverable: {filename}'
    source_config = tmp_path / 'source.json'
    source_config.write_text(json.dumps({'data': asdict(data), 'train': asdict(config)}))
    for name, value in {
        'IPYTHONDIR': str(tmp_path / 'ipython'), 'JUPYTER_RUNTIME_DIR': str(tmp_path / 'jupyter'),
        'SPNO_PROJECT_ROOT': str(project), 'SPNO_SOURCE_ROOT': str(root),
        'SPNO_OUTPUT_ROOT': str(tmp_path / 'output'), 'SPNO_SOURCE_CONFIG': str(source_config),
        'SPNO_OPTIONS': json.dumps({'seeds': [3], 'lambdas': [0.0], 'fractions': [1.0],
                                   'sigmas': [0.0], 'gammas': [0.0], 'grid': 16, 'noise': [0.0],
                                   'refinements': [3, 6]})}.items():
        monkeypatch.setenv(name, value)
    notebook = nbformat.read(path, as_version=4)
    # Execute every shipped cell, with an optimizer guard inside the actual kernel.
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import torch\ndef forbidden_training(*args, **kwargs):\n"
        "    raise AssertionError('Notebook attempted implicit training')\n"
        "torch.optim.AdamW.step = forbidden_training"))
    executed = NotebookClient(notebook, timeout=120, kernel_name='python3',
                              resources={'metadata': {'path': str(project)}}).execute()
    assert not any(output.output_type == 'error' for cell in executed.cells
                   if cell.cell_type == 'code' for output in cell.outputs)


def test_phase6_evaluation_notebook_runs_arms_incrementally(budget_bound_standalone, tmp_path, monkeypatch):
    from dataclasses import asdict
    source, data, _ = budget_bound_standalone
    project = Path(__file__).resolve().parents[1]
    source_config = tmp_path / 'source.json'
    source_config.write_text(json.dumps({'data': asdict(data)}))
    output = tmp_path / 'output'
    # A reference computed exactly as on Windows, so the G5 comparison really runs.
    from scripts.run_phase6 import _model_from_checkpoint, dispersion_arms
    from spno.checkpoints import load_checkpoint_payload
    from spno.artifacts import file_digest
    references = tmp_path / 'g5-references'
    references.mkdir()
    checkpoint = next((source / 'checkpoints' / 'phase6').rglob('C1-seed0.pt'))
    payload = load_checkpoint_payload(checkpoint)
    model = _model_from_checkpoint(payload, data, expected_name='C1')
    model.load_state_dict(payload.state_dict, strict=True)
    experiments = dispersion_arms({'base/C1': model.eval()}, data.domain, data)
    (references / 'base-C1-seed0.json').write_text(json.dumps({
        'tag': 'base/C1', 'checkpoint_sha256': file_digest(checkpoint), 'experiments': experiments}, default=float))
    monkeypatch.setenv('SPNO_G5_REFERENCE_DIR', str(references))
    for name, value in {
        'IPYTHONDIR': str(tmp_path / 'ipython'), 'JUPYTER_RUNTIME_DIR': str(tmp_path / 'jupyter'),
        'SPNO_PROJECT_ROOT': str(project), 'SPNO_SOURCE_ROOT': str(source),
        'SPNO_OUTPUT_ROOT': str(output), 'SPNO_SOURCE_CONFIG': str(source_config),
        'SPNO_OPTIONS': json.dumps({'arms': ['G5a', 'G6a'], 'expected_references': 1})}.items():
        monkeypatch.setenv(name, value)
    notebook = nbformat.read(project / 'notebooks' / '06_phase6_evaluation.ipynb', as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        "import torch\ndef forbidden_training(*args, **kwargs):\n"
        "    raise AssertionError('Notebook attempted implicit training')\n"
        "torch.optim.AdamW.step = forbidden_training"))
    for _ in range(2):  # the second pass must skip finished arms
        executed = NotebookClient(notebook, timeout=300, kernel_name='python3',
                                  resources={'metadata': {'path': str(project)}}).execute()
        errors = [o for c in executed.cells if c.cell_type == 'code' for o in c.outputs if o.output_type == 'error']
        assert not errors, errors[0].get('evalue')
    markers = {p.stem: json.loads(p.read_text()) for p in (output / 'phase6-arms').glob('*.json')}
    assert set(markers) == {'G5a', 'G6a'}
    assert all(m['exploratory'] and m['identifier'].endswith('-budget-bound') for m in markers.values())
    log = ''.join(o.get('text', '') for c in executed.cells if c.cell_type == 'code' for o in c.outputs)
    assert 'G6a  already done' in log
    assert 'compared 1 checkpoints' in log and 'PASS' in log
