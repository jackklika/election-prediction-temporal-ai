"""Local-disk object store, for tests that do not need a server.

Not a production backend — no versioning, no concurrent-writer guarantees. It
exists so unit tests can exercise the archive path without Docker, and so the
ObjectStore protocol has a second implementation keeping it honest about not
leaking S3 concepts.
"""

from __future__ import annotations

from pathlib import Path

from predictelection.storage.base import (
    ObjectNotFound,
    StoredObject,
    content_key,
    content_sha256,
)


class FilesystemObjectStore:
    scheme = "file"

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root).resolve()

    def uri_for(self, sha256: str) -> str:
        return (self._root / content_key(sha256)).as_uri()

    def _path(self, uri: str) -> Path:
        prefix = self._root.as_uri()
        if not uri.startswith(f"{prefix}/"):
            raise ValueError(f"{uri!r} is not stored under {prefix}")
        return Path(self._root / uri[len(prefix) + 1 :])

    def put(self, data: bytes, *, media_type: str | None = None) -> StoredObject:
        del media_type  # nothing on disk records it
        digest = content_sha256(data)
        uri = self.uri_for(digest)
        path = self._path(uri)
        if path.exists():
            return StoredObject(
                uri=uri, sha256=digest, byte_length=len(data), already_present=True
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-then-rename, so a reader never sees a half-written object
        temporary = path.with_suffix(".partial")
        temporary.write_bytes(data)
        temporary.replace(path)
        return StoredObject(uri=uri, sha256=digest, byte_length=len(data))

    def get(self, uri: str) -> bytes:
        path = self._path(uri)
        if not path.exists():
            raise ObjectNotFound(uri)
        return path.read_bytes()

    def exists(self, uri: str) -> bool:
        return self._path(uri).exists()
