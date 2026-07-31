"""A content-addressed blob store, narrow enough to swap backends.

Artifact rows record a storage_uri, and the URI scheme is what identifies the
backend — s3:// today, gs:// or file:// without touching a caller. Keeping the
interface to put/get/exists rather than a general filesystem is what makes that
swap cheap: none of the operations a POSIX API would add have an obvious meaning
on every provider.

Objects are addressed by the sha256 of their content, so a put is idempotent by
construction: re-archiving the same bytes is a no-op that returns the same URI,
which is exactly the behaviour a retried Temporal activity needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256 as _sha256
from typing import Protocol, runtime_checkable


def content_sha256(data: bytes) -> str:
    return _sha256(data).hexdigest()


def content_key(sha256: str) -> str:
    """Object key for a digest, sharded so no prefix grows unbounded.

    Flat prefixes are slow to list and, on some providers, a throughput limit.
    """

    if len(sha256) != 64 or not all(c in "0123456789abcdef" for c in sha256):
        raise ValueError(f"not a lowercase sha256 digest: {sha256!r}")
    return f"sha256/{sha256[:2]}/{sha256[2:4]}/{sha256}"


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Where bytes ended up, in the terms an Artifact row needs."""

    uri: str
    sha256: str
    byte_length: int
    version_id: str | None = None
    """Provider version, when the bucket has versioning enabled."""

    already_present: bool = False
    """True when the digest was already stored, so nothing was uploaded."""


@runtime_checkable
class ObjectStore(Protocol):
    """The contract an artifact backend has to satisfy."""

    @property
    def scheme(self) -> str:
        """URI scheme this store reads and writes, e.g. "s3"."""
        ...

    def uri_for(self, sha256: str) -> str:
        """The URI a digest maps to, without touching the backend."""
        ...

    def put(
        self,
        data: bytes,
        *,
        media_type: str | None = None,
    ) -> StoredObject:
        """Store bytes under their digest, returning where they live."""
        ...

    def get(self, uri: str) -> bytes:
        """Read back an object previously stored here."""
        ...

    def exists(self, uri: str) -> bool: ...


class ObjectNotFound(LookupError):
    """The URI does not resolve to an object in this store."""


def parse_uri(uri: str, *, expected_scheme: str) -> tuple[str, str]:
    """Split a bucket-style URI into (container, key), rejecting other backends.

    A store must refuse URIs it does not own, otherwise a mixed-backend database
    silently reads the wrong place.
    """

    scheme, separator, remainder = uri.partition("://")
    if not separator or scheme != expected_scheme:
        raise ValueError(f"{uri!r} is not a {expected_scheme}:// URI")
    container, _, key = remainder.partition("/")
    if not container or not key:
        raise ValueError(f"{uri!r} is missing a container or key")
    return container, key
