import json
import os
import secrets
import socket
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from optima.arenas import referee_source_digest
from optima.source_release import (
    RELEASE_MANIFEST,
    REQUIRED_RUNTIME_FILES,
    RefereeReleaseError,
    build_referee_source_release,
    verify_referee_source_release,
)


REPO = Path(__file__).resolve().parents[1]


def _minimal_source(root: Path) -> Path:
    source = root / "source"
    for relative in sorted(REQUIRED_RUNTIME_FILES):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    (source / "optima" / "referee_release.py").write_text(
        '"""release identity receipt"""\n'
        'APPROVED_REFEREE_SOURCE_DIGEST = (\n'
        '    "sha256:' + "0" * 64 + '"\n)\n'
        'APPROVED_REFEREE_TREE_DIGEST = (\n'
        '    "sha256:' + "0" * 64 + '"\n)\n',
        encoding="utf-8",
    )
    (source / "optima" / "runtime_extra.py").write_text("VALUE = 1\n")
    return source.resolve()


def _publication(root: Path) -> Path:
    path = root / "published"
    path.mkdir(mode=0o700, parents=True)
    return path.resolve()


def _make_directory_writable(path: Path) -> None:
    os.chmod(path, stat.S_IMODE(path.stat().st_mode) | 0o200)


def _freeze_directory(path: Path) -> None:
    os.chmod(path, 0o555)


def test_real_checkout_build_is_minimal_and_arena_digest_compatible(tmp_path):
    release = build_referee_source_release(REPO, _publication(tmp_path))

    assert release.referee_source_digest == referee_source_digest(REPO / "optima")
    assert referee_source_digest(release.root / "optima") == release.referee_source_digest
    assert REQUIRED_RUNTIME_FILES.issubset({entry.path for entry in release.files})
    assert (release.root / "optima_kernels/override.py").is_file()
    assert (
        release.root
        / "optima_kernels/collective/fused_ar_rmsnorm_sm103.cu"
    ).is_file()
    assert not (release.root / ".git").exists()
    assert not (release.root / "tests").exists()
    assert not list(release.root.rglob("*.pyc"))
    assert not list(release.root.rglob("__pycache__"))


def test_build_copies_only_python_and_exact_package_data_and_reuses_identity(tmp_path):
    source = _minimal_source(tmp_path)
    data = source / "optima" / "data" / "policy.json"
    data.parent.mkdir()
    data.write_text('{"version":1}\n')
    cache = source / "optima" / "__pycache__"
    cache.mkdir()
    (cache / "runtime_extra.cpython-311.pyc").write_bytes(b"ignored")
    publication = _publication(tmp_path)

    first = build_referee_source_release(
        source,
        publication,
        package_data=("optima/data/policy.json",),
    )
    second = build_referee_source_release(
        source,
        publication,
        package_data=("optima/data/policy.json",),
    )

    assert first == second
    assert first.root.name == "sha256-" + first.tree_digest.removeprefix("sha256:")
    assert (first.root / "optima/data/policy.json").read_bytes() == data.read_bytes()
    assert not (first.root / "optima/__pycache__").exists()
    assert [path for path in publication.iterdir() if path.name.startswith(".referee-build-")] == []
    for path in first.root.rglob("*"):
        assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(RefereeReleaseError, match="package-data policy is not approved"):
        verify_referee_source_release(first.root)
    assert verify_referee_source_release(
        first.root, package_data=("optima/data/policy.json",)
    ) == first


def test_manifest_is_canonical_and_covers_exact_published_tree(tmp_path):
    release = build_referee_source_release(
        _minimal_source(tmp_path), _publication(tmp_path)
    )
    raw = release.manifest_path.read_bytes()
    manifest = json.loads(raw)

    assert raw == json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    assert manifest["tree_digest"] == release.tree_digest
    assert manifest["referee_source_digest"] == release.referee_source_digest
    expected = {entry.path for entry in release.files} | {RELEASE_MANIFEST}
    actual = {
        path.relative_to(release.root).as_posix()
        for path in release.root.rglob("*")
        if path.is_file()
    }
    assert actual == expected


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("notes.txt", "unapproved referee package file"),
        ("native.so", "unapproved referee package file"),
        (".hidden", "unsafe path component"),
    ],
)
def test_unapproved_regular_package_file_is_rejected(tmp_path, name, message):
    source = _minimal_source(tmp_path)
    (source / "optima" / name).write_bytes(b"not runtime source")

    with pytest.raises(RefereeReleaseError, match=message):
        build_referee_source_release(source, _publication(tmp_path))


def test_package_data_must_be_exact_present_and_below_optima(tmp_path):
    source = _minimal_source(tmp_path)
    data = source / "optima" / "policy.json"
    data.write_text("{}\n")
    publication = _publication(tmp_path)

    with pytest.raises(RefereeReleaseError, match="unapproved referee package file"):
        build_referee_source_release(source, publication)
    with pytest.raises(RefereeReleaseError, match="non-Python path below a runtime package"):
        build_referee_source_release(source, publication, package_data=("../policy.json",))
    with pytest.raises(RefereeReleaseError, match="approved package_data files are missing"):
        data.unlink()
        build_referee_source_release(
            source, publication, package_data=("optima/policy.json",)
        )


def test_symlinked_source_file_and_directory_are_rejected(tmp_path):
    source = _minimal_source(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("SECRET = True\n")
    link = source / "optima" / "linked.py"
    link.symlink_to(outside)

    with pytest.raises(RefereeReleaseError, match="symlink is forbidden"):
        build_referee_source_release(source, _publication(tmp_path))

    link.unlink()
    directory_link = source / "optima" / "linked_dir"
    directory_link.symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(RefereeReleaseError, match="symlink is forbidden"):
        build_referee_source_release(source, _publication(tmp_path / "second"))


@pytest.mark.parametrize("kind", ["fifo", "socket"])
def test_fifo_and_unix_socket_in_source_are_rejected(tmp_path, kind):
    source = _minimal_source(tmp_path)
    special = source / "optima" / f"special_{kind}"
    listener = None
    short_alias = None
    if kind == "fifo":
        os.mkfifo(special)
    else:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        short_alias = Path("/tmp") / f"optima-release-{secrets.token_hex(6)}"
        short_alias.symlink_to(source / "optima", target_is_directory=True)
        listener.bind(str(short_alias / special.name))
    try:
        with pytest.raises(RefereeReleaseError, match="non-regular filesystem object"):
            build_referee_source_release(source, _publication(tmp_path))
    finally:
        if listener is not None:
            listener.close()
        if short_alias is not None:
            short_alias.unlink()


def test_special_object_in_ignored_bytecode_cache_is_still_rejected(tmp_path):
    source = _minimal_source(tmp_path)
    cache = source / "optima" / "__pycache__"
    cache.mkdir()
    os.mkfifo(cache / "not-bytecode.pyc")

    with pytest.raises(RefereeReleaseError, match="unapproved entry in referee bytecode cache"):
        build_referee_source_release(source, _publication(tmp_path))


def test_missing_worker_or_prebuild_source_is_rejected(tmp_path):
    source = _minimal_source(tmp_path)
    (source / "optima/eval/oci_worker.py").unlink()

    with pytest.raises(RefereeReleaseError, match="required OCI runtime files"):
        build_referee_source_release(source, _publication(tmp_path))


def test_verifier_rejects_mutation_writable_files_and_unexpected_entries(tmp_path):
    source = _minimal_source(tmp_path)
    first = build_referee_source_release(source, _publication(tmp_path))
    target = first.root / "optima/runtime_extra.py"
    os.chmod(target, 0o644)
    with pytest.raises(RefereeReleaseError, match="release file is writable"):
        verify_referee_source_release(first.root)

    target.write_text("VALUE = 2\n")
    os.chmod(target, 0o444)
    with pytest.raises(RefereeReleaseError, match="identity mismatch"):
        verify_referee_source_release(first.root)

    second_source = _minimal_source(tmp_path / "second")
    second = build_referee_source_release(second_source, _publication(tmp_path / "second"))
    package = second.root / "optima"
    _make_directory_writable(package)
    extra = package / "unexpected.py"
    extra.write_text("VALUE = 3\n")
    os.chmod(extra, 0o444)
    _freeze_directory(package)
    with pytest.raises(RefereeReleaseError, match="differs from manifest"):
        verify_referee_source_release(second.root)


def test_verifier_rejects_symlink_even_when_target_is_regular(tmp_path):
    release = build_referee_source_release(
        _minimal_source(tmp_path), _publication(tmp_path)
    )
    target = release.root / "optima/runtime_extra.py"
    package = target.parent
    _make_directory_writable(package)
    target.unlink()
    target.symlink_to("__init__.py")
    _freeze_directory(package)

    with pytest.raises(RefereeReleaseError, match="symlink is forbidden"):
        verify_referee_source_release(release.root)


def test_changed_source_publishes_new_content_addressed_tree(tmp_path):
    source = _minimal_source(tmp_path)
    publication = _publication(tmp_path)
    first = build_referee_source_release(source, publication)
    (source / "optima/runtime_extra.py").write_text("VALUE = 2\n")
    second = build_referee_source_release(source, publication)

    assert first.root != second.root
    assert first.tree_digest != second.tree_digest
    assert first.referee_source_digest != second.referee_source_digest
    assert {path.name for path in publication.iterdir()} == {
        first.root.name,
        second.root.name,
    }


def test_full_tree_digest_covers_validator_kernel_python_and_cuda(tmp_path):
    source = _minimal_source(tmp_path)
    publication = _publication(tmp_path)
    first = build_referee_source_release(source, publication)

    override = source / "optima_kernels/override.py"
    override.write_text("POINTS = {}\n")
    second = build_referee_source_release(source, publication)
    assert second.tree_digest != first.tree_digest
    # Compatibility identity intentionally matches arenas.referee_source_digest,
    # which covers optima/.  OCI integration must pin the stronger tree_digest too.
    assert second.referee_source_digest == first.referee_source_digest

    cuda = source / "optima_kernels/collective/fused_ar_rmsnorm_sm103.cu"
    cuda.write_text("// changed validator kernel\n")
    third = build_referee_source_release(source, publication)
    assert third.tree_digest not in {first.tree_digest, second.tree_digest}
    assert third.referee_source_digest == first.referee_source_digest


def test_concurrent_publication_has_one_complete_atomic_result(tmp_path):
    source = _minimal_source(tmp_path)
    publication = _publication(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as executor:
        releases = list(
            executor.map(
                lambda _: build_referee_source_release(source, publication),
                range(8),
            )
        )

    assert len({release.root for release in releases}) == 1
    assert len({release.tree_digest for release in releases}) == 1
    assert len(list(publication.iterdir())) == 1
    verify_referee_source_release(releases[0].root)


def test_builder_can_require_both_registered_identities(tmp_path):
    source = _minimal_source(tmp_path)
    publication = _publication(tmp_path)
    release = build_referee_source_release(source, publication)

    assert build_referee_source_release(
        source,
        publication,
        expected_tree_digest=release.tree_digest,
        expected_referee_source_digest=release.referee_source_digest,
    ) == release
    with pytest.raises(RefereeReleaseError, match="tree identity mismatch"):
        build_referee_source_release(
            source,
            publication,
            expected_tree_digest="sha256:" + "0" * 64,
        )
    with pytest.raises(RefereeReleaseError, match="source identity mismatch"):
        build_referee_source_release(
            source,
            publication,
            expected_referee_source_digest="sha256:" + "0" * 64,
        )


def test_referee_receipt_literals_are_normalized_but_enforced(tmp_path):
    source = _minimal_source(tmp_path)
    first = build_referee_source_release(source, _publication(tmp_path / "first"))

    receipt_source = source / "optima" / "referee_release.py"
    text = receipt_source.read_text()
    text = text.replace("sha256:" + "0" * 64, first.referee_source_digest, 1)
    text = text.replace("sha256:" + "0" * 64, first.tree_digest, 1)
    receipt_source.write_text(text)
    second = build_referee_source_release(source, _publication(tmp_path / "second"))
    assert second.tree_digest == first.tree_digest
    assert second.referee_source_digest == first.referee_source_digest

    # Non-receipt bytes remain covered. Return literals to placeholders so a new
    # identity can be computed deliberately rather than accepting stale receipts.
    text = receipt_source.read_text()
    text = text.replace(first.referee_source_digest, "sha256:" + "0" * 64, 1)
    text = text.replace(first.tree_digest, "sha256:" + "0" * 64, 1)
    receipt_source.write_text(text + "# policy byte changed\n")
    third = build_referee_source_release(source, _publication(tmp_path / "third"))
    assert third.tree_digest != first.tree_digest

    # Receipt literals are normalized for the row hash, then independently checked.
    _make_directory_writable(first.root)
    receipt = first.root / "optima" / "referee_release.py"
    os.chmod(receipt, 0o644)
    receipt.write_text(
        receipt.read_text().replace(first.tree_digest, "sha256:" + "f" * 64, 1)
    )
    os.chmod(receipt, 0o444)
    _freeze_directory(first.root)
    with pytest.raises(RefereeReleaseError, match="approved digest literals"):
        verify_referee_source_release(first.root)


def test_release_verifier_rejects_same_byte_hardlink_alias(tmp_path):
    release = build_referee_source_release(
        _minimal_source(tmp_path), _publication(tmp_path)
    )
    victim = release.root / "optima" / "runtime_extra.py"
    external = tmp_path / "external.py"
    external.write_bytes(victim.read_bytes())
    _make_directory_writable(release.root)
    _make_directory_writable(victim.parent)
    victim.unlink()
    os.link(external, victim)
    os.chmod(victim, 0o444)
    _freeze_directory(victim.parent)
    _freeze_directory(release.root)
    with pytest.raises(RefereeReleaseError, match="hardlinked"):
        verify_referee_source_release(release.root)


def test_limits_and_untrusted_publication_root_fail_closed(tmp_path):
    source = _minimal_source(tmp_path)
    publication = _publication(tmp_path)
    with pytest.raises(RefereeReleaseError, match="exceeds 1 files"):
        build_referee_source_release(source, publication, max_files=1)
    with pytest.raises(RefereeReleaseError, match="exceeds 1 bytes"):
        build_referee_source_release(source, publication, max_file_bytes=1)

    os.chmod(publication, 0o777)
    with pytest.raises(RefereeReleaseError, match="group/world writable"):
        build_referee_source_release(source, publication)


def test_publish_root_and_source_package_must_not_be_symlinks(tmp_path):
    source = _minimal_source(tmp_path)
    real_publication = _publication(tmp_path)
    publication_link = tmp_path / "publication-link"
    publication_link.symlink_to(real_publication, target_is_directory=True)
    with pytest.raises(RefereeReleaseError, match="publish_root must not be a symlink"):
        build_referee_source_release(source, publication_link)

    real_package = source / "optima"
    moved_package = source / "real-optima"
    real_package.rename(moved_package)
    real_package.symlink_to(moved_package, target_is_directory=True)
    with pytest.raises(RefereeReleaseError, match="must be a real directory"):
        build_referee_source_release(source, real_publication)
