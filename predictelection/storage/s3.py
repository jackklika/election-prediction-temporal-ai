"""S3-compatible object store, used for both MinIO and AWS.

One implementation covers both: MinIO speaks the S3 API, so it differs only by
endpoint_url. That keeps the integration tests exercising the same code path that
runs in production, which is the point of testing against MinIO rather than a
mock.
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from predictelection.clients._base_config import ConfigBase
from predictelection.storage.base import (
    ObjectNotFound,
    StoredObject,
    content_key,
    content_sha256,
    parse_uri,
)


class S3Config(ConfigBase):
    model_config = SettingsConfigDict(env_prefix="s3_")

    bucket: str = "predictelection"
    endpoint_url: str | None = None
    """Set for MinIO; leave unset for real AWS so boto resolves the region."""

    region: str = "us-east-1"
    access_key_id: str | None = None
    secret_access_key: str | None = None

    @model_validator(mode="after")
    def _credentials_are_paired(self):
        if (self.access_key_id is None) != (self.secret_access_key is None):
            raise ValueError("set both S3 credentials or neither")
        return self


LOCAL_MINIO_DEFAULTS = {
    "endpoint_url": "http://localhost:9000",
    "access_key_id": "minioadmin",
    "secret_access_key": "minioadmin",
}


def local_minio_config(**overrides: Any) -> S3Config:
    """Config for the docker-compose MinIO, for dev scripts and integration tests.

    Environment wins over the local fallback, so the same code runs against real
    S3 by setting S3_ENDPOINT_URL (or unsetting it) and real credentials — no
    branch on "am I in dev".
    """

    fallback = {
        name: value
        for name, value in LOCAL_MINIO_DEFAULTS.items()
        if f"S3_{name.upper()}" not in os.environ
    }
    return S3Config(**(fallback | overrides))


class S3ObjectStore:
    """Content-addressed storage on any S3-compatible endpoint."""

    scheme = "s3"

    def __init__(self, config: S3Config | None = None, *, client: Any = None) -> None:
        self._config = config or S3Config()
        self._client = client or boto3.client(
            "s3",
            endpoint_url=self._config.endpoint_url,
            region_name=self._config.region,
            aws_access_key_id=self._config.access_key_id,
            aws_secret_access_key=self._config.secret_access_key,
            # MinIO needs path addressing; AWS accepts it too.
            config=BotoConfig(s3={"addressing_style": "path"}),
        )

    @property
    def bucket(self) -> str:
        return self._config.bucket

    def uri_for(self, sha256: str) -> str:
        return f"s3://{self.bucket}/{content_key(sha256)}"

    def ensure_bucket(self) -> None:
        """Create the bucket if missing.

        Deliberately explicit rather than called from put: an application that
        creates buckets on demand will happily write to a typo in production.
        Dev and test setup call this; the write path never does.
        """

        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self._client.create_bucket(Bucket=self.bucket)

    def put(self, data: bytes, *, media_type: str | None = None) -> StoredObject:
        digest = content_sha256(data)
        uri = self.uri_for(digest)
        stored = StoredObject(
            uri=uri, sha256=digest, byte_length=len(data), already_present=True
        )

        existing = self._head(uri)
        if existing is not None:
            return StoredObject(
                uri=uri,
                sha256=digest,
                byte_length=len(data),
                version_id=existing.get("VersionId"),
                already_present=True,
            )

        extra: dict[str, Any] = {}
        if media_type is not None:
            extra["ContentType"] = media_type
        response = self._client.put_object(
            Bucket=self.bucket,
            Key=content_key(digest),
            Body=data,
            **extra,
        )
        return StoredObject(
            uri=stored.uri,
            sha256=digest,
            byte_length=len(data),
            version_id=response.get("VersionId"),
            already_present=False,
        )

    def get(self, uri: str) -> bytes:
        bucket, key = parse_uri(uri, expected_scheme=self.scheme)
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except ClientError as error:
            raise ObjectNotFound(uri) from error
        return response["Body"].read()

    def exists(self, uri: str) -> bool:
        return self._head(uri) is not None

    def _head(self, uri: str) -> dict[str, Any] | None:
        bucket, key = parse_uri(uri, expected_scheme=self.scheme)
        try:
            return self._client.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return None
