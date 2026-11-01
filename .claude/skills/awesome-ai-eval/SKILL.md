---
name: awesome-ai-eval
description: Use when working in the awesome-ai-eval repository or with its list - choosing an AI evaluation framework, benchmark, or platform from the list; adding or proposing an entry; reviewing a pull request; checking links; moving entries to the archive file; or updating the sindresorhus/awesome submission.
---

# Awesome AI Eval

A curated list of tools, benchmarks, platforms, and guides for evaluating language models, RAG pipelines, and AI agents. This file is the full procedure for any agent working with the list; `AGENTS.md` at the repository root is the short version. The commands need only `git`, `gh`, `curl`, and `pnpm`.

Read the part that matches the task:

- Picking a tool for someone: **Use**
- Adding an entry or opening a pull request: **Add**
- Reviewing pull requests, checking links, archiving: **Maintain**
- Anything else in this repository: **Repo**

## Use

`README.md` is the list. Read the section that matches what is being tested, then recommend from it.

| The person wants to | Section |
| --- | --- |
| Score model outputs in code or CI | Frameworks |
| Test retrieval or RAG answers | Retrieval |
| Attack a model to find failures | Adversarial |
| Trace and monitor a running system | Self-Hosted, Managed, Cloud |
| Compare models on a standard test | General, Long-Context, Specialized, Agents, Multimodal, Safety |
| See current model rankings | Leaderboards |
| Learn how evaluation should be done | Papers, Guides |

To find the right tool for someone:

1. Establish what they are testing: a model, a RAG pipeline, or an agent.
2. Establish when they test it: once against a standard benchmark, on every code change, or continuously in production.
3. Establish their limits: open source or paid, self-hosted or managed, and their programming language.
4. Read the matching section in full. Entries are alphabetical, so the first one is not the best one.
5. Shortlist two or three entries that fit, and say in one sentence each why it fits.
6. Open each shortlisted repository. Confirm it is not archived and still does what the entry says.

Do not recommend from `archive.md`.

## Add

`CONTRIBUTING.md` holds the rules. Read it first; it is short. In brief:

1. Search `README.md` and `archive.md` for the name and URL. Stop if it is listed.
2. Check the project's main purpose is evaluating AI systems, the link loads without a login, and the repository is not archived:

   ```sh
   gh api repos/OWNER/REPO --jq '{archived, stargazers_count, pushed_at, description}'
   ```

3. Check the bar: on topic, a working link, and something useful for evaluating AI systems. New or small projects are welcome; decline only what is out of scope, broken, or duplicated.
4. Add one line in alphabetical order:

   ```markdown
   - [Name](https://example.com) - What it is and what it does.
   ```

5. Change only that file. No badges, images, bold, or emoji.
6. Run `pnpm install` then `pnpm lint`. If the linter reports "Awesome list must reside in a valid git repository", use the command under Audit.
7. Title the pull request `Add Name`. One entry per pull request.

Write the description from the project's own README. State what it does in concrete nouns. Leave out marketing words and counts that will change.

## Maintain

### Review

For each open pull request:

1. Read the diff. It should change one line of one file.
2. Check the scope and placement rules in `CONTRIBUTING.md`.
3. On topic and meets the bar: accept into `README.md`. Out of scope, the link fails, or a duplicate: decline, stating what happened in one plain sentence.
4. Fix formatting yourself; a formatting mistake is not a reason to decline.
5. Accepting: if the pull request adds one correctly formatted line to the right file, merge it. Before merging, remove extra links, counts that will change, and marketing words, and move the line into alphabetical order; push those fixes to the contributor's branch when allowed, otherwise apply them after merging. If the pull request touches other files or uses an old format, add the entry by hand, then close it with a reply that says where the entry went.

Reply plainly. State the decision in the first sentence, then the rule, then the fact, then what would change the answer:

```text
Declining this pull request.
Rule: the link must load without a login.
Fact: https://github.com/OWNER/REPO returns 404 as of YYYY-MM-DD.
If the repository becomes public, reopen this and it will be accepted.
```

Do not merge, close, or comment on a pull request until the owner says to. Draft the reply and show it.

### Audit

Format:

```sh
pnpm lint
```

On a local branch with no remote set, the linter reports "Awesome list must reside in a valid git repository". Supply the remote for that run:

```sh
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=branch.$(git branch --show-current).remote GIT_CONFIG_VALUE_0=origin pnpm lint
```

Repositories that are archived, renamed, missing, or quiet:

```sh
grep -o '](https://github.com/[^/)]*/[^/)#]*' README.md | sed 's#](https://github.com/##' | sort -u |
  while read -r repo; do
    gh api "repos/$repo" --jq "[\"$repo\", .full_name, (.archived|tostring), .pushed_at[0:10]] | @tsv"
  done
```

A `full_name` that differs from the first column means the repository moved. An error line means the repository is gone or the request failed; check which before acting.

Other links that fail:

```sh
grep -o '](https\?://[^)]*)' README.md | sed 's/^](//; s/)$//' | grep -v '^https://github.com/' | sort -u |
  while read -r url; do
    printf '%s %s\n' "$(curl -s -o /dev/null -L --max-time 25 -A 'Mozilla/5.0' -w '%{http_code}' "$url")" "$url"
  done | grep -v '^200 '
```

A 403 from a large site usually means it blocks scripts; report it to the owner. A 200 can still be wrong: read the page title and check it names the project and still describes an evaluation product. Parked domains, acquisitions, and companies that changed direction all return 200.

arXiv links that point at the wrong paper:

```sh
ids=$(grep -o 'arxiv.org/abs/[0-9.]*' README.md | sed 's#.*/##' | paste -sd, -)
curl -s "https://export.arxiv.org/api/query?id_list=$ids&max_results=100" | grep -E '<id>http|<title>'
```

Check that each returned title names the entry or the paper that introduced it. ColBench, for example, comes from the SWEET-RL paper.

### Archive

Move an entry to `archive.md` when its repository is archived, when a tool has had no commits for 12 months, or when the product is gone or has changed into something else. Write the reason after the description. A finished benchmark or dataset stays on the list.

### Rules

- Section headings are one word. Contents has one level of nesting and never lists Contributing.
- No `####` headings, no `---` rules, no introductory sentences under headings.
- No badges, star counts, or tickers in `README.md`. The upstream maintainer rejected per-entry star badges as noise that goes stale.
- The only animation is the glint on the shades in `assets/robot-shades.svg` (header) and `assets/footer.svg`. The two glints are identical: change both or neither. Each SVG carries its own `<style>`; keep the `prefers-reduced-motion` rule.
- No license text or license section in `README.md`.
- The first text line under the title describes the subject, not the list. It must not start with "A curated list", and no paragraph follows it.

## Repo

| File | Purpose |
| --- | --- |
| `README.md` | The curated list. |
| `archive.md` | Entries that are archived, unmaintained, or discontinued, each with a reason. |
| `CONTRIBUTING.md` | The rules for scope, placement, and format. |
| `AGENTS.md` | Short instructions for any AI agent; points here. `CLAUDE.md` imports it. |
| `.claude/skills/awesome-ai-eval/SKILL.md` | This file. The only committed part of `.claude/`. |
| `assets/robot-shades.svg` | Header logo, with a glint across the shades. Its outline is a mid grey so it shows on dark and light themes. |
| `assets/footer.svg` | Footer: the same robot on a fading hairline, with the same glint. |
| `package.json` | Exists only to run `awesome-lint` through `pnpm lint`. |
| `LICENSE` | CC0. Never mention it in `README.md`. |

Conventions:

- Work on a branch. The owner commits, pushes, and merges.
- Commit messages are one line. Read `git log --oneline -10` and match it.
- Write plainly. Formal and clean: active voice, concrete nouns, no filler words.

Upstream:

- The list is submitted to the official index in pull request 3923 of `sindresorhus/awesome`, from the fork `Vvkmnn/awesome`, branch `add-eval`.
- Check its state with `gh pr view 3923 --repo sindresorhus/awesome --comments`.
- Never close that pull request. Upstream has new pull requests disabled, so it must be updated in place.
- The entry line belongs at the bottom of the Machine Learning sub-list in the upstream `readme.md`.
- The upstream checklist says "Only has awesome items" and "Is not AI-generated", and requires the owner to review four other pull requests. Those reviews and the final wording are the owner's to write.
