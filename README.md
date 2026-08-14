# ElectAlpha

[![CI](https://github.com/jackklika/election-prediction-temporal-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/jackklika/election-prediction-temporal-ai/actions/workflows/ci.yml)

This project is a tool to use to collect citable infromation about elections, with the goal to run analysis on these facts. It interacts with data sources to collect facts about a race and candidate, and uses an agent to research and populate a knowledge graph.

For example, we can break elections down into Candidates that run in Races which have Outcomes, participate with other Candidates in Events like Debates where the candidates produce Speech, have Polls run against them. These Races take places in Geographies that can be represented by polygons, and Populations vote for Candidates which results in Outcomes.

This is the "map" part of ["The Map is not the Territory"](https://en.wikipedia.org/wiki/Map%E2%80%93territory_relation). The further goal is to use this knowledge graph as a broad map which can then be enriched by private data or expert experience to create electoral theories that can be backtested.

# Technical decisions

- **Facts sources**: We gather facts mainly by an agent doing web search, instead of data scources like votehub, AP news. This is mainly because I'm cheap, but also because I will assume that journalists will always write stories about these races.
- **Database**: Postgres stores ontology models for claims, entities, and links between them. It was chosen because Postgres is familiar to me and is a solid foundation for an MVP. If this was a sparse KG, it may not be as good of a fit, but we are trying to keep it dense. If needed, we can extend it with Apache AGE, or migrate the data to a more comprehensive graph database.
- **Workflow execution**: Temporal is used for workflow execution. Pydantic AI Agents have native integration with it, the durable execution model is useful for non-deterministic LLM inferrence and handling failures, and it provides a great interface for interacting with long-running workflows.
- **Artifact/blob storage**: S3 or Google Cloud Storage Objects are used because where else would we cheaply store scraped artifacts? Minio is used during development for integration testing, even though it is archived, because it seems stable and is only used locally for testing.
- **Language**: Python is the standard for these kinds of AI projects. Speed is not a top concern. If it becomes a concern, we can use marturin and write some rust code that is called by python. 

# Goals

The goal is to have an agent which can do research on elections to populate a facts or "claims" knowledge graph.

Populating the knowledge graph will be by running specific workflows for different entities. For example to find which debates they took part in, polls, etc.

Then we can correlate things or enrich them. For example, we could get the transcript of debates via yt-dlp, or see crosstabs across different elections in the same geography. We can also create entities for races and create polygons for the regions, for example.

This is all in a shared ontology or knowledge base, defined in the sql schemas.

# Target Structure

- Scrape election/political data into "facts" database using agents
- Allow human-in-the-loop review of facts, including fact corrections, which can help agents understand where they went wrong before
- Enable visualization or questions to be made about this data, with scraping filling in the gaps
- Extract more abstract claims from personal writing or opinion writing, synthesizing human intuition with raw facts

# Inspiration
- [Wikontic: Constructing Wikidata-Aligned, Ontology-Aware Knowledge Graphs with Large Language Models](https://aclanthology.org/2026.eacl-long.388.pdf): This is helping me understand how to work with wikidata, inserting triplets into a knowledge graph, and performing entity resolution.

# AI disclosure

**All human text (non-code text) like this README or all the commit messages, are written personally without llm assistance or generation.**

Claude Code has written a lot of code, including models, business logic, and infrastructure. 

Initially I wanted to make this AI-free. My AGENTS.md file said "do not allow any AI generated code to enter this repo". But it was too much typing, and felt too slow when I have a good high-level vision of what to create here. My fingers started hurting, and I was moving too slow.

For example, the kalshi client configuration is essentially a 1-1 mapping of our domain models to kalshi responses, and kalshi endpoints are derived from its API. I'm not typing that all out when I can just point it to a openapi yaml file.

Or creating simple CRUD objects like the review queue are trivial and extremely testable. It is effective to give the agent requirements, it implements, I test, I give feedback, and it fixes, and it gets to a very good place.

And for the sql models -- I have little experience modeling knowledge about events, like "Facts", "claims", or "ontology-aware knowledge graphs". But I understand what the paper is saying, and want it represented in sql. So having high-capability agents parse the research, present it to me, and being able to synthesize it into code... isn't that compelling? 

Time is too valuable. I have a family I'd rather spend time with, LLMs are here to stay, and make a side project like this so much easier.

The original commits were fully personally-typed code, and I think that helped me think more carefully about the core of the application, especially understanding the pydnatic temporal llm agent construction and configuration.
