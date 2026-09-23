from pathlib import Path
import zipfile

import pytest


def artifact_tree(root, *, full=True):
    checkpoint = root / 'checkpoints/phase6/run/A-seed0.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'checkpoint')
    for split in ('train', 'val', 'test') if full else ('val', 'test'):
        path = root / 'data/hash' / f'{split}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(split.encode())
    return root


def test_discovers_phase6_local_extraction_instead_of_assuming_drive_copy(tmp_path):
    from spno.artifacts import resolve_standalone_source
    source = artifact_tree(tmp_path / 'content/pin/spno/results/phase6-standalone-artifacts')
    assert resolve_standalone_source(search_roots=[tmp_path / 'content']) == source


def test_parent_selection_resolves_nested_standalone_root(tmp_path):
    from spno.artifacts import resolve_standalone_source
    source = artifact_tree(tmp_path / 'transfer/spno/results/phase6-standalone-artifacts')
    assert resolve_standalone_source(tmp_path / 'transfer') == source


def test_prefers_complete_copy_to_evaluation_only_transfer(tmp_path):
    from spno.artifacts import resolve_standalone_source
    artifact_tree(tmp_path / 'eval-only', full=False)
    full = artifact_tree(tmp_path / 'full')
    assert resolve_standalone_source(search_roots=[tmp_path]) == full


def test_evaluation_only_transfer_reports_missing_training_data(tmp_path):
    from spno.artifacts import resolve_standalone_source
    source = artifact_tree(tmp_path / 'eval-only', full=False)
    with pytest.raises(FileNotFoundError, match='train.pt'):
        resolve_standalone_source(source)


def test_multiple_complete_runs_require_selection(tmp_path):
    from spno.artifacts import resolve_standalone_source
    artifact_tree(tmp_path / 'one')
    artifact_tree(tmp_path / 'two')
    with pytest.raises(ValueError, match='Multiple'):
        resolve_standalone_source(search_roots=[tmp_path])


def test_explicit_invalid_source_is_not_silently_replaced(tmp_path):
    from spno.artifacts import resolve_standalone_source
    artifact_tree(tmp_path / 'good')
    with pytest.raises(FileNotFoundError, match='SOURCE_ROOT'):
        resolve_standalone_source(tmp_path / 'wrong', search_roots=[tmp_path])


def test_discovers_and_extracts_full_zip_without_modifying_archive(tmp_path):
    from spno.artifacts import resolve_standalone_source
    archive = tmp_path / 'drive/phase6-standalone-artifacts-demo.zip'
    archive.parent.mkdir()
    prefix = 'spno/results/phase6-standalone-artifacts/'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr(prefix + 'checkpoints/phase6/run/A-seed0.pt', b'checkpoint')
        for split in ('train', 'val', 'test'):
            handle.writestr(prefix + f'data/hash/{split}.pt', split.encode())
    original = archive.read_bytes()
    result = resolve_standalone_source(search_roots=[archive.parent], extraction_root=tmp_path / 'extracted')
    assert (result / 'data/hash/train.pt').read_bytes() == b'train'
    assert archive.read_bytes() == original
    assert resolve_standalone_source(archive, extraction_root=tmp_path / 'extracted') == result


def test_zip_traversal_is_rejected_before_extraction(tmp_path):
    from spno.artifacts import resolve_standalone_source
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('../escape.txt', 'bad')
        handle.writestr('checkpoints/phase6/run/A-seed0.pt', b'checkpoint')
        for split in ('train', 'val', 'test'):
            handle.writestr(f'data/hash/{split}.pt', split.encode())
    with pytest.raises(ValueError, match='Unsafe'):
        resolve_standalone_source(archive, extraction_root=tmp_path / 'extracted')
    assert not (tmp_path / 'escape.txt').exists()


def test_missing_source_reports_archive_and_expected_layout(tmp_path):
    from spno.artifacts import resolve_standalone_source
    with pytest.raises(FileNotFoundError, match='phase6-standalone-artifacts.*zip'):
        resolve_standalone_source(search_roots=[tmp_path])


def test_known_windows_archive_requires_its_recorded_checksum(tmp_path):
    from spno.artifacts import resolve_standalone_source
    archive = tmp_path / 'phase6-standalone-artifacts-bd4e108527-K0.zip'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('fake', 'not the verified Windows transfer')
    with pytest.raises(ValueError, match='checksum mismatch'):
        resolve_standalone_source(archive, extraction_root=tmp_path / 'extracted')
