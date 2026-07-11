from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.referee_runtime import resolve_referee_runtime
from optima.source_release import (
    RELEASE_MANIFEST,
    RefereeReleaseError,
    build_referee_source_release,
)


REPO = Path(__file__).resolve().parents[1]


def _identity(tmp_path):
    release = build_referee_source_release(REPO, tmp_path / "identity")
    return release.tree_digest, release.referee_source_digest


def test_checkout_is_materialized_in_disjoint_private_cache(tmp_path):
    tree, source = _identity(tmp_path)
    release = resolve_referee_runtime(
        REPO,
        tmp_path / "published",
        expected_tree_digest=tree,
        expected_referee_source_digest=source,
    )

    assert release.root.parent == (tmp_path / "published").resolve()
    assert (release.root / RELEASE_MANIFEST).is_file()
    assert release.tree_digest == tree
    assert release.referee_source_digest == source


def test_verified_release_is_reused_without_publication_access(tmp_path):
    release = build_referee_source_release(REPO, tmp_path / "published")
    reused = resolve_referee_runtime(
        release.root,
        tmp_path / "does-not-need-to-exist",
        expected_tree_digest=release.tree_digest,
        expected_referee_source_digest=release.referee_source_digest,
    )

    assert reused.root == release.root
    assert not (tmp_path / "does-not-need-to-exist").exists()


def test_invalid_release_is_not_reinterpreted_as_checkout(tmp_path):
    release = build_referee_source_release(REPO, tmp_path / "published")
    manifest = release.root / RELEASE_MANIFEST
    manifest.chmod(0o600)
    manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RefereeReleaseError):
        resolve_referee_runtime(
            release.root,
            tmp_path / "fallback",
            expected_tree_digest=release.tree_digest,
            expected_referee_source_digest=release.referee_source_digest,
        )
    assert not (tmp_path / "fallback").exists()


@pytest.mark.parametrize("relation", ["same", "inside_source", "contains_source"])
def test_checkout_and_publication_must_be_disjoint(tmp_path, relation):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "optima").mkdir()
    (source / "optima_kernels").mkdir()
    if relation == "same":
        publication = source
    elif relation == "inside_source":
        publication = source / "release-cache"
    else:
        publication = tmp_path

    with pytest.raises(RefereeReleaseError, match="disjoint"):
        resolve_referee_runtime(
            source,
            publication,
            expected_tree_digest="sha256:" + "1" * 64,
            expected_referee_source_digest="sha256:" + "2" * 64,
        )


def test_crownable_cli_resolves_release_before_profile(monkeypatch, tmp_path):
    from optima import cli
    import optima.referee_runtime as runtime

    observed = {}
    release_root = tmp_path / "release"
    release_root.mkdir()

    def fake_resolve(source_root, publish_root, **expected):
        observed.update(
            source_root=Path(source_root), publish_root=Path(publish_root), **expected
        )
        return SimpleNamespace(root=release_root)

    monkeypatch.setattr(runtime, "resolve_referee_runtime", fake_resolve)
    args = SimpleNamespace(
        oci_source_dir=str(tmp_path / "checkout"),
        oci_release_root=str(tmp_path / "cache"),
    )
    arena = SimpleNamespace(
        referee_tree_digest="sha256:" + "1" * 64,
        referee_source_digest="sha256:" + "2" * 64,
    )

    assert cli._registered_referee_source(args, arena) == release_root
    assert observed == {
        "source_root": tmp_path / "checkout",
        "publish_root": tmp_path / "cache",
        "expected_tree_digest": arena.referee_tree_digest,
        "expected_referee_source_digest": arena.referee_source_digest,
    }
