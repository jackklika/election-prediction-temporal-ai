.PHONY: test test-db db-up db-down lint worker research demo migrate migration stamp \
        review review-next

# Skips the tests that need PostgreSQL or MinIO.
test:
	uv run pytest

# Runs everything with the services REQUIRED: unreachable PostgreSQL or MinIO
# fails rather than skips, so the CHECK constraints, partial indexes, and the
# real archive path are actually exercised. This is the run that matters before
# trusting a schema or ingestion change.
test-db: db-up
	uv run pytest --require-postgres

db-up:
	docker compose up -d --wait

db-down:
	docker compose down

# Bring a real database up to the current schema. This is the only path that
# works against a database with data in it: create_all cannot alter an existing
# CHECK constraint or enum, so a new predicate kind applied through it is
# silently a no-op.
migrate: db-up
	uv run alembic upgrade head

# Adopt Alembic on a database that predates it — one built by create_all, which
# already has every table the baseline would create. Records the revision
# without running it. Only correct when the schema already matches the models;
# `make test-db` proves that.
stamp: db-up
	uv run alembic stamp head

# Autogenerate a revision from whatever the models say now, then READ IT.
# Autogenerate does not detect every change — CHECK constraint edits in
# particular come out empty and have to be written by hand.
#   make migration MESSAGE="add contest-key namespace"
migration:
	@test -n "$(MESSAGE)" || (echo 'usage: make migration MESSAGE="what changed"'; exit 1)
	uv run alembic revision --autogenerate -m "$(MESSAGE)"

# Runs the workflows and activities. Leave this going in its own terminal; it is
# what actually does the research, so a run survives `make research` exiting.
#
# The S3 vars point at compose's MinIO. They are set here rather than defaulted
# in code because S3Config's no-endpoint default is real AWS, which is what
# production should get; pointing at localhost is the local-only exception.
worker: db-up
	S3_ENDPOINT_URL=http://localhost:9000 \
	S3_ACCESS_KEY_ID=minioadmin \
	S3_SECRET_ACCESS_KEY=minioadmin \
	uv run python -m predictelection.worker.worker

# Start a real research run. Needs `make worker` alongside it, and spends
# Anthropic credit. Watch it at http://localhost:8080
#   make research SUBJECT="Abdul El-Sayed"
import-ocd: db-up
	S3_ENDPOINT_URL=http://localhost:9000 \
	S3_ACCESS_KEY_ID=minioadmin \
	S3_SECRET_ACCESS_KEY=minioadmin \
	uv run python -m predictelection.importers.run ocd

import-fec: db-up
	S3_ENDPOINT_URL=http://localhost:9000 \
	S3_ACCESS_KEY_ID=minioadmin \
	S3_SECRET_ACCESS_KEY=minioadmin \
	uv run python -m predictelection.importers.run fec --cycle "$(or $(CYCLE),2026)"

#   make research SUBJECT="Michigan governor 2026" KIND=structure
research:
	@test -n "$(SUBJECT)" || (echo 'usage: make research SUBJECT="Abdul El-Sayed" [KIND=debates|structure]'; exit 1)
	uv run python -m predictelection.workflows.trigger "$(SUBJECT)" --kind "$(or $(KIND),debates)"

# What ingestion could not decide for itself. Postgres only: review reads and
# writes the graph, and never fetches or asks a model.
#   make review               what is waiting
#   make review-next          answer them one at a time
#   make review ARGS="show 6d6d24ba"
review: db-up
	uv run python -m predictelection.review $(or $(ARGS),list)

review-next: db-up
	uv run python -m predictelection.review next

# The whole ingestion path on a fixed debate, with no agent and no API calls.
demo: db-up
	uv run python -m predictelection.research.demo

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check
