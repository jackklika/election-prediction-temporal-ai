.PHONY: test test-db db-up db-down lint worker research demo

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
research:
	@test -n "$(SUBJECT)" || (echo 'usage: make research SUBJECT="Abdul El-Sayed"'; exit 1)
	uv run python -m predictelection.workflows.trigger "$(SUBJECT)"

# The whole ingestion path on a fixed debate, with no agent and no API calls.
demo: db-up
	uv run python -m predictelection.research.demo

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check
