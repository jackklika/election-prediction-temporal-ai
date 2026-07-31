"""Object store behaviour that must hold for every backend.

Runs on the filesystem store so it needs no Docker, but the assertions are
protocol-level: anything swapped in later — GCS, Azure — has to satisfy these,
which is what keeps the S3 vocabulary from leaking into callers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from predictelection.storage import (
    FilesystemObjectStore,
    ObjectNotFound,
    ObjectStore,
    content_key,
    content_sha256,
)


@pytest.fixture
def store(tmp_path: Path) -> FilesystemObjectStore:
    return FilesystemObjectStore(tmp_path / "artifacts")


def test_the_filesystem_store_satisfies_the_protocol(
    store: FilesystemObjectStore,
) -> None:
    assert isinstance(store, ObjectStore)


def test_storing_is_addressed_by_content(store: FilesystemObjectStore) -> None:
    stored = store.put(b"debate transcript", media_type="text/plain")

    assert stored.sha256 == content_sha256(b"debate transcript")
    assert stored.byte_length == len(b"debate transcript")
    assert stored.uri == store.uri_for(stored.sha256)
    assert store.get(stored.uri) == b"debate transcript"


def test_storing_the_same_bytes_twice_is_a_no_op(store: FilesystemObjectStore) -> None:
    """What makes a retried archive activity safe."""

    first = store.put(b"same bytes")
    second = store.put(b"same bytes")

    assert first.uri == second.uri
    assert first.already_present is False
    assert second.already_present is True


def test_different_bytes_get_different_uris(store: FilesystemObjectStore) -> None:
    assert store.put(b"one").uri != store.put(b"two").uri


def test_a_missing_object_raises_rather_than_returning_empty(
    store: FilesystemObjectStore,
) -> None:
    missing = store.uri_for("a" * 64)
    assert store.exists(missing) is False
    with pytest.raises(ObjectNotFound):
        store.get(missing)


def test_a_store_refuses_uris_it_does_not_own(store: FilesystemObjectStore) -> None:
    with pytest.raises(ValueError):
        store.get(f"s3://bucket/{content_key('b' * 64)}")


def test_keys_are_sharded_so_no_prefix_grows_unbounded() -> None:
    digest = "ab" + "c" * 62
    assert content_key(digest) == f"sha256/ab/cc/{digest}"


@pytest.mark.parametrize("bad", ["", "xyz", "A" * 64, "a" * 63])
def test_only_lowercase_sha256_digests_are_accepted(bad: str) -> None:
    with pytest.raises(ValueError):
        content_key(bad)
