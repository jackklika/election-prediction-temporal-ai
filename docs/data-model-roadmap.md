# Data model roadmap

What to change before scraping at volume, and why. Written after two real agent
runs against the live graph exposed where the model bends, and after reviewing
how comparable knowledge graphs have been built over the last few years.

The organising question is **which changes get expensive once data exists**.
Most of what follows is additive and can wait. Four things are not.

---

## North star: what we are building, formally

A knowledge graph is defined as ([Survey][survey], Def. 1):

> **G = { E, R, T, F_k }**

| symbol | is | here |
|---|---|---|
| **E** | entities and concepts | `entity` — people, offices, contests, elections, events, jurisdictions, parties, markets |
| **R** | relations | `predicate` + `predicate_version` — the versioned, seeded catalog |
| **T** | the set of factual triples | `claim` |
| **F_k** | *background knowledge that constrains which facts are admissible* | `predicate_subject_kind`, `predicate_object_kind`, `temporal_mode`, and the value schemas |

A binary fact is **(h, r, t)** with `h, t ∈ E` and `r ∈ R`. An n-ary fact is
**(e₁, …, eₙ, r)**. The constraint that matters is:

> **T ⊂ F_k({E, R})**

Every fact must satisfy the ontology. `F_k` is not decoration — it is what makes
this a knowledge graph rather than, in the survey's words, *"merely a data
graph"*. Construction itself is defined as `f : D × f_k(D) → G` and is
*"usually unable to continue without background knowledge"*, which is why the
predicate catalog is a prerequisite rather than an afterthought.

### Our one extension: nothing enters T without provenance

The definition says nothing about where facts come from. This project's whole
premise is that they must be citable, so we add a fifth element:

> **for every t ∈ T, at least one assertion citing archived, re-retrievable bytes**

That is `ClaimAssertion` → `EvidenceAnchor` → `SourceSnapshot` → `Artifact`. A
fact nobody can re-check is the one thing the model exists to prevent.

### The invariants, in plain language

Copy-pasteable as agent context, and the test any schema change should pass:

```text
1. EVERY FACT IS A CLAIM about entities, using a predicate from the catalog.
   Facts are not free text. If no predicate fits, the predicate is missing —
   propose one rather than forcing the fact into a near-match.

2. NO CLAIM WITHOUT EVIDENCE. Every claim cites archived bytes, a locator
   inside them, and who asserted it. A fact you cannot cite must be omitted,
   not guessed.

3. THE ONTOLOGY IS CHECKED, NOT ASSUMED. A predicate declares which entity
   kinds it accepts. A claim that violates it is stored AND flagged for
   review — never silently dropped, never silently accepted.

4. TIME IS EXPLICIT, INCLUDING ITS PRECISION. Say when a fact holds and how
   precisely the source said it. Never claim more precision than the source
   gave: a date is a date, not a midnight timestamp.

5. FACTS ARE IMMUTABLE. Corrections supersede; nothing is edited or deleted.
   The graph must be able to answer "what did we believe on date X".

6. IDENTITY IS RESOLVED, NOT ASSUMED. The same real thing must land on one
   entity. Prefer an external identifier over a name, and an existing entity
   over a new one.
```

Each maps to something built: `predicate_version`; the assertion chain;
`check_claim_ontology` + `ReviewTask`; `TimePrecision`; `Immutable` +
`ClaimSupersession`; `resolve_entity_mention` + `EntityRedirect`.

---

## Where we are

The model records a *claim* (a proposition) separately from an *assertion*
(someone asserting it from archived evidence). That separation, plus
content-addressed archiving and append-only review, is working and should not be
disturbed — the research below says so more strongly than expected.

Two runs of `find_debates` on the same subject produced:

- entity resolution that was **exact** for people, contests, and jurisdictions
  (0 duplicates across runs), and
- **11 event entities for 6 real debates**, because debate titles have no
  canonical form and the agent re-phrased them between runs
  (`(WOOD TV8` vs `(WOOD-TV`, a `First`/`Second` prefix, `U.S. Senate Democratic`
  vs `Democratic U.S. Senate`).

Resolution reliability tracks how canonical a name is. Short canonical names
("Michigan", "Abdul El-Sayed") resolve perfectly on exact match; long descriptive
titles never will.

### Coverage against the README

| README concept | Status |
|---|---|
| Candidate | ✅ `PERSON` + `candidate_in` — correctly a role, not an entity kind |
| Race | ⚠️ `CONTEST` exists; no primary/general distinction, no party |
| **Outcome** | ❌ **missing entirely** — backtesting is impossible without it |
| Event / Debate | ✅ |
| Speech | ✅ mostly — `public_statement` + `VideoEvidenceLocator` + `ArtifactDerivation` already express "utterance cited into a transcript"; needs the yt-dlp pipeline, not new schema |
| Poll | ✅ |
| Geography | ⚠️ `JURISDICTION` exists, no polygons; PostGIS installed and unused |
| Population | ❌ missing; genuinely additive |

---

## What the research says

### 0. The framing: acquisition → refinement → evolution

Zhong et al. survey 300+ KG construction methods and organise the field into
three stages: **knowledge acquisition** (entity discovery, entity typing, entity
linking, coreference resolution, relation extraction), **knowledge refinement**
(KG completion, knowledge fusion), and **knowledge evolution** (conditional
knowledge, temporal dynamics) ([Survey][survey]).

We do acquisition and a thin slice of fusion. We do **no refinement** — nothing
infers missing edges or reconciles across sources — and evolution only insofar as
supersession exists. That is the right order of operations, and worth knowing as
the map of what we are deliberately not doing yet.

Three things in it bear directly on decisions below.

**The formal definition already includes n-ary relations and an ontology
constraint.** A knowledge graph is `G = {E, R, T, F_k}` where *"a standard binary
fact is a triple (h,r,t)"* and — explicitly — *"an n-ary relation triple will be
formed as (e₁, …, eₙ, r)"*. `F_k` is *"a set function representing the background
knowledge that constrains potential facts"*, with `T ⊂ F_k({E, R})`. Construction
itself is defined as *"usually unable to continue without background knowledge"*.

So ternary claims are not an extension of the model — they are in the definition.
And `predicate_subject_kind`/`predicate_object_kind` *are* `F_k`. Which is why the
survey can say: *"if a KG system does not organize nodes and edges with background
knowledge about concepts, **it is merely a data graph**"*.

**Our duplication problem is a named classic defect.** Traditional extraction
systems are faulted as *"insubstantial: they do not create or distinguish entities
from different expressions, which **prevents knowledge aggregation**"*. Eleven
event entities for six debates is that defect, verbatim.

**The classic pipeline links entities *before* extracting relations** — *"first
discovers and links conceptual entities, resolves coreference mentions, then
extracts relationships among entities"*. Ours does the opposite: the agent emits
entities and relations together and we resolve afterward. That is the structural
reason it re-describes rather than re-uses, and the strongest argument for the
read tools in step 5.

### 1. The claim/assertion split is the nanopublication pattern

Nanopublications separate every fact into three graphs: the **Assertion** (the
claim), the **Provenance** (what it was derived from), and the **PublicationInfo**
(who generated it, when) ([nanopub guidelines][nanopub], [CEUR][nanoprov]).
That is `Claim` / `EvidenceAnchor`+`SourceSnapshot` / `ResearchRun`, arrived at
independently.

More useful: PROV-O, *"the most widely adopted conceptual model for representing
provenance"*, is noted to **lack representation of supporting and contradicting
evidence and reliability scores** ([CEUR][nanoprov]). Our `EvidenceStance`
(`supports`/`contradicts`/`mentions`) and `ClaimAssertion.confidence` cover
exactly that gap. This is the one area where the model is ahead of the standards,
and it is worth not regressing.

### 2. Validity intervals, atomic facts, and explicit invalidation are the state of the art

ATOM (2026) constructs temporal knowledge graphs from LLM output using **validity
intervals rather than single timestamps**, decomposes text into **"atomic
facts" — minimal, indivisible units** — and explicitly invalidates superseded
facts *"without simply duplicating outdated entries"*. It identifies the failure
of prior systems as *"struggling with temporal consistency, often failing to
properly invalidate outdated information"* ([ATOM][atom]).

All three already exist here: `valid_from`/`valid_to` with precision,
`ingest_debate` emitting one claim per assertion rather than one per debate, and
`ClaimSupersession`. Keep them.

### 3. Ternary claims are mainstream; reification is the expensive option

For attaching a value to a relationship — `(Murphy) assessed (Crowley) as
(weakest)` — Hogan et al. compare four RDF strategies and find standard
reification roughly **doubles dataset size** and queries poorly, since
reconstructing one statement requires matching several patterns
([Reifying RDF][hogan]). The compact alternatives are RDF-star, which exists
precisely to *"add descriptions to edges — scores, weights, temporal aspects and
provenance"* ([Ontotext][ontotext]), and property graphs with native edge
properties.

Caveat to carry forward: **property-graph edge properties are literals only**
([CEUR][ceur]). A JSONB value has the same limit, so *entity-valued* qualifiers
still need their own predicate. The pressing cases are literal.

### 4. Entity resolution is a cascade, and naive string matching is a known failure

The production pattern is **Rules → ML → LLM**, cascading to balance cost,
latency, and accuracy: embeddings block candidates into groups, then an LLM
matches and merges ([Graphlet][graphlet], [Medium][er-scale]). Practice also
holds that entity resolution is *"a platform capability rather than a one-time
project phase"* and that *"most enterprise builds stall at entity resolution and
ontology alignment, not extraction"* ([Atlan][atlan], [Zylos][zylos]).

Most directly: in *"popular Graph-based RAG systems like LightRAG and MS
GraphRAG, simple string matching is currently used for deduplication, though this
misses entities with the same semantic meaning but different forms"*
([Graphlet][graphlet]). That is precisely our 11-for-6 result. We are at tier one
of a three-tier cascade, and the tools everyone uses stop there too.

The survey reaches the same place from the other end: *"asking appropriate users
to complete and correct knowledge graphs is the ultimate solution for obtaining
unknown facts in the open world"*, and proposes routing uncertain items to the
right kind of reviewer — field experts, organised authorities, or automated
systems ([Survey][survey]). `ReviewTask` plus `ReviewerKind` is that queue; the
open extension is routing by reviewer competence rather than a flat priority.

### 5. Identifiers are plural, and schemes do get deprecated

Confirmed, and it changes the earlier recommendation. Practice is that **"all
identifiers continue to exist, rather than eliminating one or more when records
are merged"**, and that persistent identifiers keep *"metadata managed separately
from the identified artifact"* ([DPC][dpc], [ORCID][orcid]). Schemes genuinely
retire: `eduPersonTargetedID` was marked deprecated in 2020 and scheduled to
become obsolete, with a recommended successor ([UK federation][ukfed]). Many
schemes coexist by design — DOI, ORCID, ROR, ISBN, IGSN.

So OCD-IDs are **one namespace among several**, not the answer. The model must
not bet on any single scheme.

---

## Should we move to Apache AGE?

**No — and not for the reason I expected.** I assumed PostgreSQL 18 would be
unsupported; it is not. AGE supports **PG 11–18** ([AGE][age]). The blocker is
architectural.

The survey's storage section is unusually direct about this. It names relational
databases — **PostgreSQL specifically** — as an established KG storage strategy,
and gives exactly two caveats: *"it can be very costly for a relational database
to handle **sparse KGs** or perform **data partition for distribution storage**"*
([Survey][survey]). Neither applies. Our graph is dense and domain-bounded, and
single-node. The known reasons to leave Postgres are reasons we do not have.

AGE stores graphs in **its own tables**. It does not expose existing relational
tables as a graph, so adopting it means *copying* claim data into a second
representation — making it a **read projection**, exactly like
`PoliticalEventProjection` already is, not a migration target.

That copy would lose what is doing the real work: 106 CHECK constraints,
immutability guards, fingerprint uniqueness, and the composite foreign keys
enforcing revision scoping. AGE has no equivalent for any of them. Cypher is also
translated into PostgreSQL function calls with parsing overhead, several
openCypher features are unsupported ([#2323][age-issue]), and a production
migration settled on *"SQL for mechanical writes, Cypher optional for
graph-shaped reads"* ([Trendyol][trendyol]). AGE is judged most valuable when an
application is *"mostly relational with a graph workload bolted on"*
([gdotv][gdotv]).

**Revisit when** a specific variable-depth traversal is painful in SQL — "everyone
within N hops of this candidate through shared events". Until such a query exists
and hurts, AGE is a second copy of the truth with weaker guarantees. Recursive
CTEs cover the fixed-depth traversals we do; `resolve_entity` is one.

---

## Status

Recommendations 1–6 are **implemented** (173 tests passing). What each turned
into is recorded below; the reasoning is unchanged, so this doubles as the record
of why the code looks the way it does.

One decision worth writing down: **the existing data was discarded rather than
migrated.** Nothing is in production, so redefining predicate v1 in place beat
bumping to v2 and re-ingesting. That licence expires the moment a scrape is worth
keeping — at which point Alembic stops being optional, because `create_all` does
not alter an existing CHECK constraint.

## The plan

Ordered by how expensive each becomes *after* data exists.

### 1. Ternary predicates

A third `PredicateTarget` where `object_id` **and** `value` are both required:

```python
PredicateSpec(slug="assessed", target_kind=PredicateTarget.QUALIFIED,
              temporal_mode=TemporalMode.REQUIRED,
              subject_kinds=(PERSON, ORGANIZATION), object_kinds=(PERSON,),
              value_model=AssessmentValue)

record_claim_from_source(predicate=spec("assessed"),
    subject_id=murphy, object_id=crowley,
    value={"rating": "weakest", "basis": "debate performance"},
    validity=Validity.on(debate_date, TimePrecision.DAY))
```

**Touches:** a branch each in `ck_claim_target_matches_payload`,
`ck_predicate_version_value_contract_matches_target`,
`PredicateSpec.__post_init__`, `_validate_claim_contract`. ~40 lines.

**Why now:** `build_claim_fingerprint` already hashes both `object_id` and
`value`, so the hash input does not change shape and **every existing claim keeps
its identity**. The survey's own definition of a knowledge graph admits n-ary
facts alongside binary triples ([Survey][survey]), so this is filling in the
model rather than stretching it. Later, anything modelled as binary that should have been ternary
needs a new predicate version and re-ingestion.

**Unlocks immediately:** `participated_in(person → event, {role})`, so moderators
stop being indistinguishable from candidates.

### 2. Outcomes

`Races have Outcomes` is central to the README and there is no way to record who
won anything. As a ternary claim it needs no new subsystem:

```python
contest_result(candidate → contest, {votes: 412_331, share: "48.7", place: 1, won: true})
```

It inherits provenance, review, and `ClaimSupersession` for the initial-count →
canvass → recount → certified progression. Polls earned their own tables because
they have real internal structure; outcomes are flat and do not.

### 3. Primary vs general, and a `PARTY` kind

Separate `CONTEST` entities, not a flag. Primaries and generals have different
dates, candidate sets, polls, and outcomes; collapsing them breaks `candidate_in`,
poll attribution, and results at once.

- `contest_stage(contest, {stage: primary|runoff|general|special})`
- `contest_party(contest → party)` — add `PARTY` to `EntityKind`; the column is
  already `VARCHAR(32)`, so it costs nothing
- `advances_to(primary → general)`

`contest_for_office` already points both at the same `OFFICE`, which is the
natural join.

### 4. Identifier strategy — revised

Earlier this said "seed OCD-IDs". That was too narrow. **OCD is one namespace of
several and may itself be deprecated**, which the research confirms is normal.

`EntityIdentifier` already supports many identifiers per entity — that part is
right. Three things are missing:

**a. A namespace registry.** A small seeded table, in the same spirit as the
predicate catalog: `namespace`, label, issuing authority, URI template,
`status` (`active` / `deprecated` / `obsolete`), `superseded_by`, and a
precedence rank. `EntityIdentifier.namespace` gets a foreign key to it.

This is exactly how schemes are retired in practice — mark the *scheme*
deprecated with a recommended successor, without touching a single identifier
row. It also catches typo'd namespaces, which the current free-text column
cannot.

**b. Nothing may hardcode a namespace.** `ScrapedEntity.as_mention` currently
bakes in `namespace="wikidata"`. Keep named fields on the scraped models —
a model fills in `wikidata_id` far more reliably than a free-form pair — but map
them through the registry rather than an if-chain, so adding `ocd_id`, `fec_id`,
or `bioguide_id` is data, not logic. Precedence then decides which wins when two
identifiers disagree, instead of the current unconditional raise.

**c. Identifiers should cite a source.** `EntityIdentifier` has no
`research_run_id` and no evidence, unlike every claim. For a system whose premise
is citability, "who says Wisconsin has this OCD ID" being unrecorded is an
inconsistency. A nullable `research_run_id` matching `ClaimAssertion` is enough.

Then seed OCD divisions as *one* high-precedence namespace for US geography.
Tier-0 resolution handles them exactly, and the same IDs are what Google Civic
and most public datasets key on — but nothing depends on OCD surviving.

### 5. Agent read tools

**Independent of 1–3 — no schema changes, so this can run in parallel.**

The agent cannot see the graph, so it re-describes rather than re-uses — the
direct cause of 11 events for 6 debates. A `find_events(kind, jurisdiction,
between)` tool lets it reuse an existing title instead of inventing a phrasing.

This is the pipeline-order fix. The classic construction pipeline **links
entities before extracting relations** ([Survey][survey]); ours currently
extracts both at once and resolves afterwards, which is precisely how "different
expressions" of one entity get created and *"knowledge aggregation"* is
prevented. Giving the agent lookup turns the second run into a linking step
against what already exists.

It attacks duplication **at the source** and belongs before fuzzy matching, which
then becomes the safety net rather than the mechanism. It also supplies the
context to resolve "last night" and "a month ago" against the publication date
and prior events, instead of guessing a false midnight.

### 6. Source authorship

`Source` is not an `Entity`, so "Murphy wrote this column" is not expressible.
For an opinion piece the author is the most important attribute, since the claims
*are* their judgments. Decide whether sources get reified as entities.

### Deliberately deferred

- **Semantic entity resolution (tiers 2–3).** The cascade's upper tiers:
  embedding blocking, then LLM adjudication, then merge proposals for review.
  Blocking on date + kind and scoring by trigram already separates our real
  duplicates (0.83–1.00) from genuinely different debates (≤0.62), so the cheap
  version works — and read tools reduce how much it must catch. `EntityRedirect`
  and `resolve_entity` already exist to apply merges retroactively without
  rewriting claims.
- **Knowledge refinement.** The survey's whole middle stage — KG completion
  (inferring missing edges) and knowledge fusion (reconciling across sources) —
  is unbuilt. Correct for now: inferring edges before the extracted ones are
  trustworthy would launder guesses into facts. Revisit once review has a
  track record.
- **Geometry.** PostGIS installed and unused; polygons on jurisdictions are
  additive.
- **Populations / electorates.** Additive.
- **Apache AGE.** See above.

---

## Sources

[survey]: https://arxiv.org/pdf/2302.05019
[nanopub]: https://nanopub.net/guidelines/working_draft/
[nanoprov]: https://ceur-ws.org/Vol-3937/paper10.pdf
[atom]: https://arxiv.org/pdf/2510.22590
[hogan]: https://aidanhogan.com/docs/reification-wikidata-rdf-sparql.pdf
[ontotext]: https://www.ontotext.com/knowledgehub/fundamentals/what-is-rdf-star/
[ceur]: https://ceur-ws.org/Vol-3279/paper2.pdf
[graphlet]: https://blog.graphlet.ai/the-rise-of-semantic-entity-resolution-45c48d5eb00a
[er-scale]: https://medium.com/@shereshevsky/entity-resolution-at-scale-deduplication-strategies-for-knowledge-graph-construction-7499a60a97c3
[atlan]: https://atlan.com/know/ai-agent/knowledge-graph/knowledge-graph-construction-for-ai/
[zylos]: https://zylos.ai/en/research/2026-02-10-knowledge-graphs-ai-systems/
[dpc]: https://www.dpconline.org/handbook/technical-solutions-and-tools/persistent-identifiers
[orcid]: https://support.orcid.org/hc/en-us/articles/360006971013-What-are-persistent-identifiers-PIDs
[ukfed]: https://www.ukfederation.org.uk/content/Documents/PersistentIdentifiers
[ocdids]: https://open-civic-data.readthedocs.io/en/latest/ocdids.html
[popolo]: https://www.popoloproject.com/
[age]: https://age.apache.org/age-manual/master/intro/setup.html
[age-issue]: https://github.com/apache/age/issues/2323
[trendyol]: https://medium.com/trendyol-tech/migrating-graph-operations-to-apache-age-from-writes-to-reads-3b8334628e1c
[gdotv]: https://gdotv.com/blog/apache-age-explained/

- [A Comprehensive Survey on Automatic Knowledge Graph Construction — Zhong et al., ACM Computing Surveys][survey]
- [Nanopublication guidelines][nanopub] · [Extending Nanopublications with Knowledge Provenance][nanoprov]
- [ATOM: Adaptive and Optimized dynamic Temporal knowledge graph construction using LLMs][atom]
- [Reifying RDF: What Works Well With Wikidata? — Hogan et al.][hogan]
- [What Is RDF-star? (Ontotext)][ontotext] · [Transforming RDF-star to Property Graphs][ceur]
- [The Rise of Semantic Entity Resolution (Graphlet)][graphlet] · [Entity Resolution at Scale][er-scale]
- [Knowledge Graph Construction for AI (Atlan)][atlan] · [Knowledge Graphs for AI Systems (Zylos)][zylos]
- [Persistent identifiers (Digital Preservation Handbook)][dpc] · [What are PIDs? (ORCID)][orcid] · [UK federation position on PIDs][ukfed]
- [Open Civic Data Identifiers][ocdids] · [Popolo][popolo]
- [Apache AGE setup and supported versions][age] · [openCypher gaps #2323][age-issue] · [Trendyol migration][trendyol] · [Apache AGE Explained][gdotv]
