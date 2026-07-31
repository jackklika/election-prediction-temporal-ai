# Political Prediction

This project is a tool to use to help predict elections. It interacts with Kalshi API to get election odds for previous or current races, and uses AI analysis and research to suggest odds and optionally place orders.

We use Temporal for its durable execution guarentees to handle workflows.

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

All human text, non-code text, like this README or all the commit messages, are written personally without llm assistance or generation.

Claude Code and Codex have written a lot of code, including models, business logic, and infrastructure. 

Initially I wanted to make this AI-free. My AGENTS.md file said "do not allow any AI generated code to enter this repo". But it was too much typing, and felt too slow when I have a good high-level vision of what to create here. My fingers started hurting!

For example, the kalshi client configuration is essentially a 1-1 mapping of our domain models to kalshi responses, and kalshi endpoints are derived from its API. I'm not typing that all out when I can just point it to a openapi yaml file.

And for the sql models -- I have little experience modeling knowledge about events, like "Facts", "claims", or "ontology-aware knowledge graphs". But I understand what the paper is saying, and want it represented in sql. So having high-capability agents parse the research, present it to me, and being able to synthesize it into code... isn't that compelling? 

Time is too valuable. I have a family I'd rather spend time with, LLMs are here to stay, and make a side project like this so much easier.

The original commits were fully personally-typed code, and I think that helped me think more carefully about the core of the application, especially understanding the pydnatic temporal llm agent construction and configuration.
