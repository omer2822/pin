from pathlib import Path
import json

import pytest

nbformat = pytest.importorskip("nbformat")
NotebookClient = pytest.importorskip("nbclient").NotebookClient

from test_workflow import standalone


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
