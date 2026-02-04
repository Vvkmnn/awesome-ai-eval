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

Each entry follows this format (badge first for visual alignment):

```
- ![](badge-url) [**Name**](https://link) - Description.
```

### Guidelines

- Keep every bullet to **one sentence** ending with a period.
- Use **bold** for the title inside the link: `[**Name**](url)`
- Place badge **before** the link for consistent visual alignment.
- Alphabetize entries within every subsection.
- Prefer HTTPS links without tracking parameters.
- Avoid marketing adjectives ("best", "awesome", "super powerful", etc.).

### Creating badges

Every entry includes a non-clickable badge before the title. Here's how to create them:

#### For GitHub repositories

Use the shields.io GitHub stars badge with `github.com` label:

```
![](https://img.shields.io/github/stars/OWNER/REPO?style=social&label=github.com)
```

**Example:** Adding `https://github.com/anthropics/evals`
1. Extract owner and repo: `anthropics` / `evals`
2. Build badge: `![](https://img.shields.io/github/stars/anthropics/evals?style=social&label=github.com)`
3. Final entry:
```
- ![](https://img.shields.io/github/stars/anthropics/evals?style=social&label=github.com) [**Anthropic Model Evals**](https://github.com/anthropics/evals) - Evaluation suite for safety, capabilities, and alignment testing.
```

#### For non-GitHub links (docs, websites, papers)

Use a static shields.io badge with the domain name:

```
![](https://img.shields.io/badge/DOMAIN-active-blue?style=social)
```

**Example:** Adding `https://docs.llamaindex.ai/en/stable/module_guides/evaluating/`
1. Extract domain: `docs.llamaindex.ai`
2. Build badge: `![](https://img.shields.io/badge/docs.llamaindex.ai-active-blue?style=social)`
3. Final entry:
```
- ![](https://img.shields.io/badge/docs.llamaindex.ai-active-blue?style=social) [**LlamaIndex Evaluation**](https://docs.llamaindex.ai/en/stable/module_guides/evaluating/) - Modules for replaying queries and comparing query engines.
```

#### Quick reference

| Link type | Badge template |
|-----------|----------------|
| GitHub repo | `![](https://img.shields.io/github/stars/OWNER/REPO?style=social&label=github.com)` |
| Documentation | `![](https://img.shields.io/badge/DOMAIN-active-blue?style=social)` |
| arXiv paper | `![](https://img.shields.io/badge/arxiv.org-active-blue?style=social)` |
| Hugging Face | `![](https://img.shields.io/badge/huggingface.co-active-blue?style=social)` |

## Pull-request checklist

Before opening a PR, confirm each item:

- [ ] The new or updated entry falls within scope and is currently maintained.
- [ ] The entry is placed in the single most relevant section and alphabetized.
- [ ] The description is under ~120 characters, single-sentence, and free of fluff.
- [ ] All sections listed in `## Contents` exist and match the heading text exactly.
- [ ] No duplicate entries exist elsewhere in the README.
- [ ] `pnpm lint` (awesome-lint) exits with no errors.

## Adding or updating entries

1. Edit `README.md` and keep sections balanced; merge or split sections sparingly.
2. Verify the project's docs and repo are reachable (no 404s) and HTTPS-only.
3. Keep descriptions concrete: mention metrics, modalities, or differentiating features.
4. Run the lint command below and fix any reported issues.
5. Update `CONTRIBUTING.md` or `README.md` if rules evolve.

### Example submission

Here's a complete example of adding a new tool to the "Core Frameworks" section:

**Before (existing entries):**
```markdown
#### Core Frameworks

- ![](https://img.shields.io/github/stars/confident-ai/deepeval?style=social&label=github.com) [**DeepEval**](https://github.com/confident-ai/deepeval) - Python unit-test style metrics for hallucination, relevance, toxicity, and bias.
- ![](https://img.shields.io/github/stars/explodinggradients/ragas?style=social&label=github.com) [**Ragas**](https://github.com/explodinggradients/ragas) - Evaluation library that grades answers, context, and grounding with pluggable scorers.
```

**After (with your new entry alphabetized):**
```markdown
#### Core Frameworks

- ![](https://img.shields.io/github/stars/confident-ai/deepeval?style=social&label=github.com) [**DeepEval**](https://github.com/confident-ai/deepeval) - Python unit-test style metrics for hallucination, relevance, toxicity, and bias.
- ![](https://img.shields.io/github/stars/org/mynewtool?style=social&label=github.com) [**MyNewTool**](https://github.com/org/mynewtool) - Brief description of what makes this tool useful for evaluation.
- ![](https://img.shields.io/github/stars/explodinggradients/ragas?style=social&label=github.com) [**Ragas**](https://github.com/explodinggradients/ragas) - Evaluation library that grades answers, context, and grounding with pluggable scorers.
```

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
