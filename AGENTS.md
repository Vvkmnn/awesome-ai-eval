# AGENTS.md

Instructions for AI agents working in this repository. The full procedure is in [`.claude/skills/awesome-ai-eval/SKILL.md`](.claude/skills/awesome-ai-eval/SKILL.md); read it before changing anything.

## Repository

A curated list of tools, benchmarks, platforms, and guides for evaluating language models, RAG pipelines, and AI agents.

- `README.md` is the list: projects that are documented, maintained, and used by others.
- `candidates.md` holds on-topic projects that are new or not yet widely used.
- `archive.md` holds entries that are archived, unmaintained, or discontinued, each with a reason.
- `CONTRIBUTING.md` holds the scope, placement, and format rules.

## Using the list

To recommend a tool, establish what is being tested (a model, a RAG pipeline, or an agent), when (once, on every change, or in production), and the constraints (open source or paid, self-hosted or managed, language). Read the matching section of `README.md` in full, shortlist two or three entries, and confirm each repository is still active. Do not recommend from `archive.md`.

## Rules

- One entry per line: `- [Name](https://example.com) - What it is and what it does.` 20 words or fewer, alphabetical within its section.
- No badges, images, bold, emoji, or counts that will change.
- Section headings are one word. No `####` headings, horizontal rules, or text under headings.
- Run `pnpm install` and `pnpm lint` after every change to `README.md`.
- Draft pull request replies and comments and show them to the owner before posting. Do not merge, close, push, or commit without the owner's approval.
- Never close or comment on `sindresorhus/awesome#3923`; it is the submission of this list to the official index.
