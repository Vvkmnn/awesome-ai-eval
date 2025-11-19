# Contributing to Awesome AI Eval

Thanks for helping improve this list! Follow the guidance below to keep the repo ready for sindresorhus/awesome review.

## Scope

This list only includes:

- Actively maintained evaluation frameworks, datasets, or platforms focused on LLMs, RAG, agents, or prompt safety.
- Publicly documented services (open-source or commercial) with clear usage instructions.
- Research papers, surveys, or datasets that directly inform evaluation methodology.

It explicitly excludes:

- Generic ML libraries without evaluation components.
- Vendor landing pages without actionable docs or public access.
- Dead, archived, or unmaintained projects.

## Entry format

- Keep every bullet to **one sentence** ending with a period.
- Use the format `- [Name](https://link) - Description.` with a standard hyphen.
- Alphabetize entries within every subsection.
- Prefer HTTPS links without tracking parameters.
- Avoid marketing adjectives (“best”, “awesome”, “super powerful”, etc.).

## Pull-request checklist

Before opening a PR, confirm each item:

- [ ] The new or updated entry falls within scope and is currently maintained.
- [ ] The entry is placed in the single most relevant section and alphabetized.
- [ ] The description is under ~120 characters, single-sentence, and free of fluff.
- [ ] All sections listed in `## Contents` exist and match the heading text exactly.
- [ ] No duplicate entries exist elsewhere in the README.
- [ ] `pnpm lint` (awesome-lint) exits with no errors.

## Adding or updating entries

- Edit `README.md` and keep sections balanced; merge or split sections sparingly.
- Verify the project’s docs and repo are reachable (no 404s) and HTTPS-only.
- Keep descriptions concrete: mention metrics, modalities, or differentiating features.
- Run the lint command below and fix any reported issues.
- Update `CONTRIBUTING.md` or `README.md` if rules evolve.

## Running awesome-lint

```sh
pnpm lint
```

The script runs `awesome-lint` locally and enforces Markdown style, heading order, link validity, and Awesome-specific rules. Fix every issue before pushing.

## Maintenance heuristics

- Re-run `pnpm lint` when dependencies change or Markdown tooling is upgraded.
- Sweep for abandoned repos (no commits for 12 months) and remove them promptly.
- Close stale PRs or issues that add low-signal items.
- Prefer PRs that include supporting evidence (blog post, release notes, or adoption proof).
- Encourage contributors to add datasets, not just tools, to keep the list balanced.
