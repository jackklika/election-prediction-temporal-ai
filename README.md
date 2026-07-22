# Political Prediction

This project is a tool to use to help predict elections. It interacts with Kalshi API to get election odds for previous or current races, and uses AI analysis and research to suggest odds and optionally place orders.

We use Temporal for its durable execution guarentees to handle workflows.

# Goals

The goal is to have something that can do research on elections to populate a facts database.

Tools will be dispatched via temporal activities within workflows. Tools can search the knowledge graph or populate it.


# AI disclosure

No AI generation has been used for any text written in this repo -- code, commit messages, and documentation is all typed out by myself, with a little copy-paste and vim functions.

This project is an exercise in manual, fingers-on-keyboard programming.

Yes, AI-generated code is valuable and useful, and coding agents save hours of toil. But I need to prove to myself I still have the chops and write some artisinal code :)

# Deployment

As a personal project, this is not intended to be put on the cloud. But
