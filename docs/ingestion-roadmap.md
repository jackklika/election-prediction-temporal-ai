# Ingestion roadmap

A handoff brief for whoever adds the next ingestion path. Written to be read
cold — you do not need any prior context beyond this file and
[`data-model-roadmap.md`](./data-model-roadmap.md).

**Where things stand:** the ingestion structure is finished and is not expected
to change again. Two agent paths (debates, race structure) and two importers
(OCD divisions, FEC candidates) write through it. Of the 19 seeded predicates,
**13 have a writer and 6 do not**: `endorsed`, `public_statement`,
`office_for_jurisdiction`, `market_for_contest`, `contest_result`, `assessed`.

The remaining work is **polls → results → endorsements**, and none of it should
require touching the activity, workflow, contract or worker layers.

---

## 1. Read this first

[`data-model-roadmap.md`](./data-model-roadmap.md) contains the north star —
`G = { E, R, T, F_k }` and six invariants. Those are not suggestions. The two that
get broken most easily by new ingestion code:

> **NO CLAIM WITHOUT EVIDENCE.** Every claim cites archived bytes, a locator
> inside them, and who asserted it. A fact you cannot cite must be omitted.
>
> **TIME IS EXPLICIT, INCLUDING ITS PRECISION.** Never claim more precision than
> the source gave: a date is a date, not a midnight timestamp.

Two rules specific to this work:

1. **Every predicate you start writing needs a test proving re-ingestion does not
   duplicate.** Use `assert_reingestion_is_idempotent` from
   `predictelection/tests/helpers.py` — it runs your ingestion twice and counts
   entities, claims, assertions, evidence anchors *and* review tasks. Counting
   claims alone is not enough; see §5.
2. **If no predicate fits what you are extracting, the predicate is missing.**
   Add one to `PREDICATE_SPECS` rather than forcing the fact into a near-match.
   Bumping an existing predicate's meaning requires a `version` bump —
   `seed_predicates` will refuse the change otherwise, by design.

---

## 2. Adding a scrape type

This is the whole procedure. If you find yourself editing anything outside these
files, the seam has leaked and the fix belongs in the seam, not in your domain.

**An agent-driven scrape** — three files plus three registry lines:

```python
# 1. predictelection/research/<domain>.py — the record and its ingestor
class ScrapedThing(ScrapedRecord):
    record_type: Literal["thing"] = "thing"
    ...

def ingest_thing(record: ScrapedThing, context: IngestContext) -> Ingestion: ...

# 2. predictelection/agents/<domain>.py
thing_agent = build_research_agent(
    name="find_things", instructions=INSTRUCTIONS, output_type=ThingFindings,
)

# 3. predictelection/workflows/<domain>.py
@workflow.defn(sandboxed=False)
class ResearchThingsWorkflow(ResearchWorkflow):
    task_type = "find_things"
    agent = thing_agent
    __pydantic_ai_agents__ = [thing_agent]

    @workflow.run
    async def run(self, request: ResearchInput) -> ResearchOutput:
        return await self.research(request)     # Temporal needs this on the subclass

    async def gather(self, request): ...        # ask the agent; that is all

# 4. predictelection/research/registry.py — the three lines
ScrapedPayload = Annotated[ScrapedDebate | ScrapedRaceStructure | ScrapedThing, ...]
INGESTORS = {..., ScrapedThing: ingest_thing}
# and predictelection/workflows/registry.py: add the workflow
```

`tests/test_research_registry.py` fails if the union and `INGESTORS` disagree, or
if the generic layers ever import a domain module again.

**An importer** — one file plus a subclass:

```python
class ThingImporter(Importer):
    name = "import_things"
    @property
    def source_url(self) -> str: ...
    def parse(self, raw: bytes) -> FilteredParse: ...      # count what you filter
    def ingest(self, row: ImportRow, context: IngestContext) -> Ingestion: ...
```

`run_import` handles archiving, the research run, per-row evidence, counting, and
not letting one bad row abort the file.

### What `IngestContext` gives you

An ingestor cannot record an unattributed claim, because the only way to write
one is through a context that already holds the snapshot, the run and the
asserter.

```python
context.resolve(EntityKind.PERSON, scraped_entity)   # or a name, or an EntityMention
context.record("candidate_in", subject_id=..., object_id=..., excerpt=...)
context.register_source(kind=SourceKind.VIDEO, canonical_url=...)
```

Pass a per-claim `locator=` whenever you know something narrower than the whole
page. Importers get this for free — `ImportRow.locator` points at the row.

### Helpers that already exist — do not rewrite these

| Need | Use |
|---|---|
| name → entity ID | `resolve_entity_mention`, `EntityMention`, `ExternalIdentifier` |
| record a fact with evidence | `IngestContext.record` → `RecordedClaim`, `ClaimOutcome` |
| archive bytes | `SourceArchive.observe` / `.artifact` / `.derive` |
| contest / office / election identity | `ContestKey`, `OfficeKey`, `ElectionKey` (§4) |
| validity intervals | `Validity.on` / `.between` / `.timeless` |
| stable keys for at-most-once writes | `idempotency_key` |
| generic dedupe by unique column | `get_or_create` |
| poll rows | `new_poll_revision`, `new_poll_option`, `new_poll_estimate`, `new_poll_average_revision` |
| corrections | `new_claim_supersession` |
| see what is already in the graph | `find_entities`, `find_events` |
| quality signal | `ontology_alignment_score` |

Everything except the keys and `IngestContext` is exported from
`predictelection.sql`; those two come from `predictelection.research`.

---

## 3. Importers vs agents

**Decide this first — it determines everything else.**

| | Importer | Agent |
|---|---|---|
| Input | structured feeds: FEC bulk, OCD, poll CSVs, certified results | unstructured text: articles, race descriptions, opinion |
| Mechanism | deterministic parsing | LLM extraction |
| Cost at volume | negligible | significant |
| Numbers | exact | can hallucinate |
| Cite with | `JsonEvidenceLocator` (row pointer, automatic) | `FullSourceLocator` or `WebEvidenceLocator` |
| Origin recorded | `RecordOrigin.IMPORT` | `RecordOrigin.MODEL` |

**Most of the remaining high-value work is an importer.** Vote counts and poll
percentages should never come from an LLM when a CSV exists.

Both paths use the same write path and produce the same `Ingestion`. An importer
archives its source file as an `Artifact` exactly like a web page, so the
provenance chain is identical.

---

## 4. Identity is derived, never named

The measured failure this exists to prevent: two runs on one subject produced
**11 event entities for 6 real debates**, because the agent re-phrased each title.
A contest is worse — "Michigan Governor 2026", "2026 Michigan gubernatorial
election" and "MI-GOV 2026" are all reasonable and all different.

So a contest, an office and an election are identified by a **derived key**, in
`research/contests.py`:

```
ocd-division/country:us/state:mi/governor/2026/primary/democratic
└──────────── division ────────┘ └office┘ cycle └stage┘ └─party─┘
```

Anything that can state the division, office, year and stage arrives at the same
string, so the FEC importer and the structure agent land on one CONTEST entity
without ever agreeing on a name. Three rules:

- **Derive, do not ask.** A model handed a key format produces plausible keys,
  and a plausible-but-wrong key mints a contest nothing else will resolve to.
  Ask for the components; compute the key in the ingestor.
- **Mark a derived key `identifiers_are_authoritative=True`.** Resolution
  normally falls back to name matching when an identifier misses, which is right
  for a *read* identifier — the OCD import saying "Michigan" should adopt the
  Michigan a debate already created. It is wrong for a *derived* key, where the
  identifier is the definition: two debates sharing a title on different days
  merged on the name, and the survivor carried both keys.
- **Events are keyed too**, on division, kind and date, plus a host when two
  share a day — `EventKey`. This is the identity the roadmap opened with (11
  event entities for 6 real debates) and it was the last kind still resolving on
  a name. Keyed only when the resolved jurisdiction carries an OCD division and
  the source gave a date to day precision; otherwise it falls back to the title,
  deliberately, because a key derived from an ID some of the time and a name the
  rest would fork on exactly the axis it exists to fix.
- **The district lives in the division**, as OCD writes it — office `us-house`,
  division `.../state:mi/cd:11`. That is what lets a contest join to the
  jurisdiction the OCD import created.
- **A primary and a general are separate CONTEST entities**, joined by
  `contest_for_office` and linked by `advances_to` — which `ContestKey.at_stage`
  derives, so nobody has to name the general. A general is never party-scoped.

---

## 5. Traps already paid for

These cost real time and API credit. Read them before debugging anything.

**A retried import re-observes, and quietly doubles every assertion.** A
`SourceSnapshot` is an *observation*, keyed on `(source, artifact, retrieved_at)`
— so archiving the same file twice legitimately makes two, and the evidence
anchor is fingerprinted over the snapshot. Claims stay deduplicated and
assertions silently double. `run_import` scopes its `ResearchRun` to the file's
**content hash** and reuses the snapshot recorded as that run's
`ResearchRunInput`; an unchanged file is a retry, a changed file is new research.
This is why the re-ingestion helper counts assertions and anchors, not claims.

**The test suite has its own database, and refuses to run anywhere else.**
`conftest.postgres_engine` drops and rebuilds every table it owns, so it uses
`predictelection_test` (created on demand) and reads `TEST_POSTGRES_URL` — never
the application's `POSTGRES_URL`. Both halves matter: it previously defaulted to
`predictelection` *and* honoured the app's variable, so running the suite while a
research run was in flight deleted that run's rows and the workflow failed on
`no research run <id>` while trying to record its own failure.

A guard refuses to start if the two URLs name the same server and database. Note
it compares them *normalized* — the first version compared raw tuples, and
because `.env` omits the port while the test URL spells out 5432, it passed
while pointed straight at production data. A safety check that silently does
nothing is worse than none; `test_conftest_guard.py` exists to keep this one
honest.

**A name-fallback merge put 154 townships into one entity.** The first real OCD
run resolved 47,039 divisions into 33,229 entities: every "Washington township"
after the first missed on its (new) identifier, fell back to the name tier,
matched the first one imported, and deposited its OCD ID on it — 5,124 entities
absorbed ~19k identifiers, and no test caught it because fixtures never
contained namesakes. The fix is in resolution: a name match that already carries
a *different* value in a namespace the mention asserts is a namesake, not the
thing itself, so it mints. Carrying *no* value in that namespace is still the
adoption case (the OCD import attaching an ID to the "Michigan" a debate
created), which must keep merging. Real reference data is the only fixture that
contains this shape — run importers against the live file before trusting
entity counts.

**An agent activity that times out retries forever, paying each time.**
`TemporalDurability` runs every model request as an activity, and Temporal's
default retry policy is *unbounded*. A `start_to_close_timeout` that is merely
too short therefore does not fail — it loops, buying the whole request again on
every attempt, and the UI shows "Attempt 4 of Unlimited" next to a healthy
heartbeat. Five minutes was not enough for a web-searching agent on a second
pass, where the prompt carries the `ALREADY RECORDED` block. `AGENT_TIMEOUT` is
now 20 minutes with `maximum_attempts=3`; keep the cap whatever you do to the
timeout. Heartbeats do not save you — the activity heartbeats right up until it
is killed, so "still working" and "about to be retried" look identical.

**`@workflow.run` cannot be inherited.** Temporal rejects the class with
"@workflow.run defined on ResearchWorkflow.run but not on the override". Each
concrete workflow declares a two-line `run` that calls `self.research(request)`.

**Autogenerate will propose dropping every PostGIS table.** The database runs
PostGIS, whose tiger geocoder installs some sixty tables Alembic reflects and
finds missing from `Base.metadata`. `include_object` in
`predictelection/sql/schema.py` filters them out; the migration tests import the
same filter, so a test on a PostGIS-free scratch database cannot pass for the
wrong reason.

**A database built before Alembic must be stamped, not migrated.** `make
migrate` on a `create_all` database fails on the first CREATE TABLE. `make
stamp` records the revision without running it — correct only when the schema
already matches, which `make test-db` proves.

**A capped lookup reads to a model as "nothing else exists".** `find_events`
returns `EntityMatches` with a `truncated` flag and orders events most-recent
first; surface it in the prompt. It was ordered by `canonical_name` with a bare
limit, so past 50 events the agent was shown an alphabetical slice unrelated to
the subject while the `ALREADY RECORDED` block still looked complete.

**Nullable JSONB stores JSON `null`, not SQL `NULL`.** SQLAlchemy's default makes
`value IS NULL` false, so any CHECK phrased that way silently rejects valid rows.
This made the entire entity half of the predicate catalog unseedable. Use
`nullable_jsonb()` for every nullable JSONB column.

**`agent.override(model=...)` does not reach inside `TemporalDurability`.** It
appears to work and silently uses the real provider — a test that meant to stub
the model spent real credit and ran 200 seconds. Inject at construction:
`build_agent(model=FunctionModel(...))`.

**A tool cannot call `workflow.execute_activity`.** `TemporalDurability` runs each
agent step *as an activity*, and an activity cannot start another. Wiring a lookup
as an agent tool hangs indefinitely rather than erroring. The workflow fetches
context and passes it into the prompt; keep it that way.

**`CodeMode` hides tools behind a sandboxed `run_code`.** Tools registered with
`tools=[...]` will not appear in the model's tool list. Its `tools=` selector
takes `(ctx, tool_def)`.

**A SQLAlchemy `Connection` is not thread-safe.** Temporal runs sync activities on
worker threads; sharing one connection produced a seven-minute hang and wrong
results. Give activities a `sessionmaker` bound to the *engine*.

**`Decimal` forks fingerprints.** Pydantic serialises it scale-preserving, so
`12.5` and `12.50` hash differently. Any `Decimal` reaching a fingerprint must be
`CanonicalDecimal`.

**`create_all` does not alter existing constraints.** Adding an enum member or a
CHECK branch to a live database silently does nothing. Alembic is set up: `make
migrate` applies, `make migration MESSAGE="..."` generates, and
`tests/test_migrations.py` fails if the models and the migrations disagree.
Autogenerate does **not** detect CHECK constraint edits — write those by hand.

**Blank environment variables shadow `.env`.** Handled in `ConfigBase` now, but
worth knowing if config reads empty.

---

## 6. Phase 2 — Polls

**Goal:** the main predictive signal. The identity and write layer is **built
and tested** (`research/polls.py`); what remains is the source parser.

**Write through `ingest_poll`, nothing else.** It layers three identities, and
each exists because a source will violate it:

- `PollKey` — pollster + contest + fieldwork end — decides *which poll this is*,
  so Wikipedia and Ballotpedia reporting the same survey land on one `Poll` row.
  Stored in `Poll.external_namespace/external_id`. No fieldwork end date means
  no key: the poll is stored unkeyed and flagged with a `ReviewTask`, never
  given an invented date.
- `payload_hash` decides whether this *reading* is new. The payload excludes
  `source_url` deliberately — two outlets printing the same numbers must hash
  identically. A *different* reading of a keyed poll becomes a second revision
  plus a `ReviewTask`: disagreement is surfaced, never averaged.
- Fuzzy checks **flag, never merge**: a trigram near-miss on the pollster name
  (pg_trgm, migration 0002) or a same-pollster same-contest poll with fieldwork
  ending within 3 days files a `ReviewTask` and proceeds.

**Pollsters resolve by slug** (`resolve_pollster`): "EPIC-MRA", "EPIC MRA" and
"EPIC/MRA" collapse via a slug alias; anything less exact goes to review.
Candidate columns are stored as verbatim option labels with `choice_entity_id`
NULL — resolving "Rogers" belongs to whoever knows the contest's candidates
(`candidate_in`), not to the poll writer.

**Next: the Wikipedia importer.** Parse raw HTML (`<table>`s vanish in
markdown/readability conversion — verified), one importer parameterized by race
page; the polling section a table sits under (D primary / R primary / general)
*is* the contest's stage and party. Validate table shape loudly: assert expected
headers, skip aggregate rows, refuse rows whose percentages misalign. A poll's
contest resolves by `ContestKey`, never by name.

**Model note:** the pollster is not the source. `PollRevision.source_snapshot_id`
points at where you read it; `pollster_id` at who conducted it. A secondary source
reporting someone else's poll is normal and the schema expects it.

**Acceptance** (first three already pass in `test_research_polls.py`):
- Importing the same reading twice writes nothing.
- A second outlet with identical numbers is a no-op; a disagreeing one is a
  second revision plus a `ReviewTask`.
- Punctuation variants of a pollster resolve to one entity; lookalikes fork
  with a `ReviewTask`.
- A crosstab estimate cannot be attached across revisions — assert the
  `IntegrityError`.

---

## 7. Phase 3 — Results

**Goal:** close the backtesting loop. Predictions cannot be scored without
outcomes, and `contest_result` currently has no writer.

**Mechanism:** importer, from certified state sources or an academic mirror.

`contest_result` is a **QUALIFIED (ternary)** predicate — subject *and* object
*and* value: `(candidate → contest, {votes, share, place, won})`. Use
`CanonicalDecimal` semantics for `share`; the value model already does, so pass a
`Decimal` and let it canonicalise. `48.7` and `48.70` must not fork.

**Counts change.** Election night → canvass → recount → certified is a
supersession chain, not an edit: record a new claim and link it with
`new_claim_supersession`, which derives the idempotency key from the two claims
so a retried correction is a no-op rather than a constraint violation.

**Acceptance:**
- A recount produces a second claim plus a supersession row; the original is
  still readable.
- `won` is set explicitly, not inferred from `place` (multi-winner contests
  exist).
- A query joining `contest_result` to `contest_for_office` returns winners by
  office — the shape backtesting will actually use. The candidate/office join in
  `test_research_structure.py` is the pattern.

---

## 8. Phase 4 — Endorsements

**Goal:** `endorsed`, which is also QUALIFIED: `(endorser → endorsee,
{strength, context})`.

**Mechanism:** agent. Endorsements live in press releases and articles, not feeds.

**Withdrawals are new claims over later intervals**, never deletions. The
`EndorsementStrength.WITHDRAWN` member exists for exactly this; the graph must
still be able to answer "who had this endorsement in June".

Endorsers are often organisations or parties, so the subject kinds include
`ORGANIZATION` and `PARTY` — resolve them properly rather than storing a name.

**Acceptance:**
- An endorsement and its later withdrawal are two claims, both retrievable, with
  distinct validity.
- Re-scraping the same endorsement from a second outlet produces one claim with
  two assertions (`ClaimOutcome.CORROBORATED`), not two claims.

---

## 9. Working agreement

- `make test-db` green before you start and before you finish. It requires
  Postgres and MinIO via `docker compose up -d`.
- `make lint` green too — `ruff format`, `ruff check`, `ty check`.
- Every phase adds tests. **Re-ingestion tests are mandatory** — see rule 1.
- Watch `ontology_alignment_score` per run. A drop means extraction is producing
  claims whose entity kinds do not match the predicate, and those are sitting in
  the review queue.
- Prefer adding a predicate over overloading one. The catalog is cheap; a
  wrong-shaped claim is not.
- Any schema change gets a migration. `create_all` is for tests only.
- **Effort is set explicitly, per agent.** `DEFAULT_EFFORT` in `agents/base.py`
  is `medium`; the debates agent pins `high`. Never inherit the API's `high`
  default by omission — that was costing 25-minute requests that hit the
  activity timeout.

  **On a research agent, effort may be a recall knob rather than just a cost
  knob** — treat that as a working hypothesis, not a result. The observation:
  Sonnet 4.6 at `high` reported six debates including a prior race, Sonnet 5 at
  `medium` reported only the three current-race ones. The model changed with the
  effort, so it is equally consistent with Sonnet 5 scoping more tightly. Pin
  `high` where coverage is the point until someone sweeps effort on a fixed
  model; leave `medium` where the question is already scoped.

  Sweep with `ANTHROPIC_EFFORT`, which deliberately beats a per-agent pin so a
  sweep cannot silently measure only the unpinned agents. A level the model does
  not support (`xhigh` before Sonnet 5 / Opus 4.7) is a 400 on every request,
  not a silent downgrade.
