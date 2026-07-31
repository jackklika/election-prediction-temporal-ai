"""Archiving a source so a claim can cite bytes that will not change.

This is the one place storage and the ontology meet. Everything a scraper reads
goes through here first, because a ClaimAssertion needs an EvidenceAnchor, an
anchor needs a SourceSnapshot, and a snapshot needs an Artifact whose bytes are
actually retained somewhere durable.

Every step is content-addressed or get-or-create, so re-archiving is a no-op and
a retried activity converges on the same rows.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from predictelection.sql import (
    Artifact,
    ArtifactDerivation,
    ArtifactDerivationKind,
    Source,
    SourceKind,
    SourceSnapshot,
    get_or_create,
)
from predictelection.storage.base import ObjectStore


class SourceArchive:
    """Records what was read, where it came from, and what it said."""

    def __init__(self, session: Session, store: ObjectStore) -> None:
        self._session = session
        self._store = store

    def source(
        self,
        *,
        kind: SourceKind,
        canonical_url: str,
        title: str | None = None,
    ) -> Source:
        source, _ = get_or_create(
            self._session,
            Source(kind=kind, canonical_url=canonical_url, title=title),
            key=Source.canonical_url == canonical_url,
        )
        return source

    def artifact(
        self,
        *,
        content: bytes,
        media_type: str | None = None,
        original_filename: str | None = None,
    ) -> Artifact:
        """Persist bytes and record them, keyed by digest in both places."""

        stored = self._store.put(content, media_type=media_type)
        artifact, _ = get_or_create(
            self._session,
            Artifact(
                sha256=stored.sha256,
                storage_uri=stored.uri,
                storage_version_id=stored.version_id,
                byte_length=stored.byte_length,
                media_type=media_type,
                original_filename=original_filename,
            ),
            key=Artifact.sha256 == stored.sha256,
        )
        return artifact

    def observe(
        self,
        *,
        kind: SourceKind,
        canonical_url: str,
        content: bytes,
        retrieved_at: datetime,
        media_type: str | None = None,
        title: str | None = None,
        published_at: datetime | None = None,
    ) -> SourceSnapshot:
        """Record one reading of a source, archiving what it said at the time.

        Re-reading a source that has not changed is still a distinct snapshot,
        because "unchanged as of today" is itself evidence.
        """

        source = self.source(kind=kind, canonical_url=canonical_url, title=title)
        artifact = self.artifact(content=content, media_type=media_type)
        snapshot, _ = get_or_create(
            self._session,
            SourceSnapshot(
                source_id=source.id,
                artifact_id=artifact.id,
                retrieved_at=retrieved_at,
                published_at=published_at,
            ),
            key=(
                (SourceSnapshot.source_id == source.id)
                & (SourceSnapshot.artifact_id == artifact.id)
                & (SourceSnapshot.retrieved_at == retrieved_at)
            ),
        )
        return snapshot

    def derive(
        self,
        *,
        parent: Artifact,
        content: bytes,
        kind: ArtifactDerivationKind,
        processor_name: str,
        processor_version: str | None = None,
        media_type: str | None = None,
        research_run_id=None,
    ) -> Artifact:
        """Record a transcript, OCR, or extracted text alongside its original.

        The derived bytes are archived in their own right, so a claim can cite
        the transcript while the lineage back to the video survives.
        """

        derived = self.artifact(content=content, media_type=media_type)
        existing = self._session.scalar(
            select(ArtifactDerivation).where(
                ArtifactDerivation.parent_artifact_id == parent.id,
                ArtifactDerivation.derived_artifact_id == derived.id,
                ArtifactDerivation.kind == kind,
            )
        )
        if existing is None:
            self._session.add(
                ArtifactDerivation(
                    parent_artifact_id=parent.id,
                    derived_artifact_id=derived.id,
                    kind=kind,
                    processor_name=processor_name,
                    processor_version=processor_version,
                    research_run_id=research_run_id,
                )
            )
            self._session.flush()
        return derived
