.PHONY: test test-db db-up db-down lint

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

lint:
	uv run ruff format --check .
	uv run ruff check .
	uv run ty check
