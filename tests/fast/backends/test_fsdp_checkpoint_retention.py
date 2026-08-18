import shutil
from types import SimpleNamespace
from unittest.mock import Mock

from miles.backends.fsdp_utils import checkpoint


def _make_checkpoint(root, step: int):
    path = root / f"iter_{step:07d}"
    path.mkdir()
    return path


def _checkpoint_names(root) -> set[str]:
    return {path.name for path in root.iterdir() if path.is_dir() and not path.is_symlink()}


def test_prune_keeps_current_and_newest_checkpoints(tmp_path):
    checkpoints = [_make_checkpoint(tmp_path, step) for step in range(1, 5)]

    checkpoint._prune_checkpoints(tmp_path, current_checkpoint=checkpoints[-1], max_checkpoints=2)

    assert _checkpoint_names(tmp_path) == {"iter_0000003", "iter_0000004"}


def test_prune_disabled_keeps_all_checkpoints(tmp_path):
    checkpoints = [_make_checkpoint(tmp_path, step) for step in range(1, 4)]

    checkpoint._prune_checkpoints(tmp_path, current_checkpoint=checkpoints[-1], max_checkpoints=None)

    assert _checkpoint_names(tmp_path) == {"iter_0000001", "iter_0000002", "iter_0000003"}


def test_prune_keeps_current_when_it_is_not_newest(tmp_path):
    checkpoints = [_make_checkpoint(tmp_path, step) for step in range(1, 5)]

    checkpoint._prune_checkpoints(tmp_path, current_checkpoint=checkpoints[1], max_checkpoints=2)

    assert _checkpoint_names(tmp_path) == {"iter_0000002", "iter_0000004"}


def test_prune_ignores_non_checkpoint_paths_and_symlinks(tmp_path):
    current = _make_checkpoint(tmp_path, 2)
    (tmp_path / "iter_invalid").mkdir()
    (tmp_path / "iter_0000001").write_text("not a directory")
    target = tmp_path / "external"
    target.mkdir()
    (tmp_path / "iter_0000003").symlink_to(target, target_is_directory=True)

    checkpoint._prune_checkpoints(tmp_path, current_checkpoint=current, max_checkpoints=1)

    assert current.exists()
    assert (tmp_path / "iter_invalid").exists()
    assert (tmp_path / "iter_0000001").is_file()
    assert (tmp_path / "iter_0000003").is_symlink()
    assert target.exists()


def test_prune_continues_after_delete_failure(tmp_path, monkeypatch, caplog):
    first = _make_checkpoint(tmp_path, 1)
    second = _make_checkpoint(tmp_path, 2)
    current = _make_checkpoint(tmp_path, 3)
    real_rmtree = shutil.rmtree

    def remove(path):
        if path == first:
            raise OSError("busy")
        real_rmtree(path)

    monkeypatch.setattr(checkpoint.shutil, "rmtree", remove)

    checkpoint._prune_checkpoints(tmp_path, current_checkpoint=current, max_checkpoints=1)

    assert first.exists()
    assert not second.exists()
    assert "Failed to remove old checkpoint" in caplog.text


def test_save_prunes_only_after_tracker_update(tmp_path, monkeypatch):
    _make_checkpoint(tmp_path, 1)
    _make_checkpoint(tmp_path, 2)
    actor = SimpleNamespace(
        args=SimpleNamespace(
            save=str(tmp_path),
            no_save_optim=True,
            fsdp_max_checkpoints_to_keep=2,
        ),
        model=object(),
        global_step=3,
        micro_step=4,
    )

    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(checkpoint.dist, "barrier", Mock())
    monkeypatch.setattr(checkpoint.dcp, "save", Mock())
    monkeypatch.setattr(checkpoint.torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(checkpoint.torch.cuda, "get_rng_state_all", lambda: [])
    real_rmtree = shutil.rmtree

    def remove(path):
        assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"
        real_rmtree(path)

    monkeypatch.setattr(checkpoint.shutil, "rmtree", remove)

    checkpoint.save(actor, iteration=2)

    assert _checkpoint_names(tmp_path) == {"iter_0000002", "iter_0000003"}
    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "3"
