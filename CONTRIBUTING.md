# Contributing

## Scope

The list covers the evaluation of AI systems: language models, RAG pipelines, and agents. Frameworks, benchmarks, datasets, leaderboards, platforms, papers, and guides are in scope. Gateways, tracing libraries, and guardrails are in scope when they include evaluation features.

## Placement

`README.md` holds every project that is on topic, has a working link, and does something useful for evaluating AI systems.

`archive.md` holds entries whose repository is archived, that are no longer maintained, or whose product is gone. A tool with no commits for 12 months moves there. A finished benchmark or dataset stays on the list.

## Format

```markdown
- [Name](https://example.com) - What it is and what it does.
```

- One line per entry, 20 words or fewer, starting with a capital letter and ending with a period.
- No badges, images, bold, or emoji.
- State what the project does. Leave out "best", "powerful", "simple", and similar words, and leave out counts that will change.
- Link the repository or the main page over HTTPS, with no tracking parameters.
- Place the entry in one section, in alphabetical order.

## Pull requests

- Add one entry per pull request, titled `Add Name`.
- Change only the file the entry belongs in.
- Check that the project is not already in `README.md` or `archive.md`.
- Run `pnpm install` and `pnpm lint` before opening the pull request.
