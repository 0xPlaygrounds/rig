# rig-coder ladder: a restartable run prompt

You are improving Rig by climbing a ladder of coding benchmarks with `rig-coder`, a coding agent built on Rig's default agent runtime (`rig-agent`). Every failure is evidence about either the agent or Rig itself; fix the cause where it lives. This file is your instructions and your memory. The **STATE** section at the bottom is yours to edit: read it first on every start, resume from it, and keep it current, so that you can be killed at any moment and restarted from this file alone. Never rewrite the instructions above STATE; add lessons to STATE › notes.

Start a session with: `Read test-support/rig-coder/LADDER.md and continue.`

## Ground rules

- **Branch and PR.** Work in the worktree of branch `feat/rig-coder` and its open PR against `0xPlaygrounds/rig` `main` (STATE › pr). Commit every kept gain on its own and push immediately. Ledger rows, STATE updates and rung reports are committed and pushed too. Never commit `runs/` or `bench/bin/`.
- **Work dirty.** No clippy, fmt, `cargo xtask` or pre-push checks. The only gate before a paid run is `cargo build -p rig-coder` plus `cargo test -p rig-coder`, and a Linux binary whose receipt matches what you are about to evaluate. CI results on the PR are information, not a gate.
- **Model** `gemini/gemini-3.8-flash`, fixed for the whole ladder so runs compare. Pass the key with `--ae GEMINI_API_KEY=$GEMINI_API_KEY`; never print it.
- **Budget.** STATE › budget is a hard cap for everything, including re-runs. Before each Harbor job, estimate its cost from the rung's observed per-trial mean (or the table), and stop with a report if known plus reserved spend would cross the cap. A trial without recorded usage is never zero: reserve it at the rung's highest observed trial cost.
- **Evidence.** Report only what ran. Re-run a failing trial alone before calling its failure real. Never edit tasks, verifiers, timeouts or the scoring scripts' pass criteria. Never read a task's `solution/` or `tests/` before its failure is analysed from the transcript, and never put task-specific text (file names, expected outputs, domain hints for one task) into the agent.
- **Contamination.** A trial whose tool calls fetch benchmark material (the dataset's repository, `solution/`, a published answer) is marked contaminated in STATE › findings and excluded from the score. The agent phase runs behind an allowlist that admits only the model provider; keep it that way.

## Where things are

- Agent: `test-support/rig-coder/src/` (`main.rs` CLI and run, `tools.rs` the six tools, `transcript.rs` the live JSONL hook, `prompt.md` the system prompt).
- Harbor adapter: `bench/rig_coder_agent.py`. Linux build: `bench/build-linux.sh` writes `bench/bin/rig-coder-linux-<arch>` and a `.build.json` receipt. Dataset snapshot with the egress policy: `bench/harbor_policy.py <dataset@version>` into `runs/datasets/`. Ledger: `bench/ledger.py summary|record|compare`, writing `bench/ledger.jsonl`.
- Per trial, in `runs/<job>/<trial>/`: `result.json` (reward, exception, tokens), `agent/transcript.jsonl` (task, assistant turns, usage, tool calls and results, and the ending), `agent/effects.json` (the replayable effect log), `agent/rig-coder.txt` (stderr), `agent/exit_code.txt`, `verifier/`.
- All commands below run from `test-support/rig-coder` with `export PYTHONPATH=$PWD`, on OrbStack Docker (`docker context ls` shows `orbstack`), with `--force-build` because task images are amd64 and this host is arm64.

## The ladder

A rung is **done** when its whole dataset has run at k=1, its ledger row exists, every failed trial is bucketed, the `harness` and `rig` fixes it motivated are kept or reverted by the keep rule, and its report is in STATE › rungs. Do not skip or subset a rung to make it pass.

| # | dataset@version | tasks | est. $/trial | why it is here |
|---|---|---:|---:|---|
| 1 | `hello-world@1.0` | 1 | 0.05 | adapter, egress policy, and ledger sanity |
| 2 | `terminal-bench-sample@2.0` | 10 | 1.0 | terminal genre, leaderboard format |
| 3 | `quixbugs@1.0` | 80 | 0.15 | single-function fixes; exercises `edit_file` |
| 4 | `humanevalfix@1.0` | 164 | 0.15 | the same across more languages |
| 5 | `aider-polyglot@1.0` | 225 | 0.25 | multi-language edits against tests |
| 6 | `livecodebench@6.0` | 100 | 0.30 | reasoning-heavy generation |
| 7 | `bigcodebench-hard-complete@1.0.0` | 145 | 0.30 | library-heavy Python |
| 8 | `terminal-bench@2.0` | 89 | 1.75 | the full terminal benchmark |
| 9 | `terminal-bench-pro@1.0` | 200 | 2.0 | harder terminal tasks |
| 10 | `swebench-verified@1.0` | 500 | 2.5 | repository-level bug fixing |

Rungs 9 and 10 will not fit the current budget; reaching them means stop and report.

## Per-rung procedure

1. **Prepare.** `cargo build -p rig-coder && cargo test -p rig-coder` from the repo root. If anything under `test-support/rig-coder/src`, `crates/`, `Cargo.toml` or `Cargo.lock` changed since the receipt's `source_head`, or the receipt says `dirty: true` for changes that are now committed, rebuild with `bench/build-linux.sh`. Snapshot: `python3 bench/harbor_policy.py <dataset@version>`. Then `harbor run -p <snapshot> -a bench.rig_coder_agent:RigCoderAgent -m gemini/gemini-3.8-flash --install-only --force-build -o runs --job-name install-<n> -y` on one task (`-i <task>`).
2. **Run** the whole rung: `harbor run -p <snapshot> -a bench.rig_coder_agent:RigCoderAgent -m gemini/gemini-3.8-flash --ae GEMINI_API_KEY=$GEMINI_API_KEY --force-build -n 4 --max-retries 0 -o runs --job-name ladder-<n>-<dataset> -y`. Set `RIG_CODER_NETWORK_PROBE=1` for a separate one-task job first on every rung and check `agent/network-probe.txt` shows every URL `DENIED`. Put the job name in STATE › in_progress_job before starting and clear it after the ledger row.
3. **Ledger** at once, before analysis: `python3 bench/ledger.py record runs/<job> --dataset <d@v> --rung <n>`. Add the job's known cost to STATE › spent.
4. **Analyse every trial.** Run `python3 bench/ledger.py summary runs/<job>`, then read each failed or errored trial's `result.json`, `agent/transcript.jsonl`, `agent/rig-coder.txt` and `verifier/` output, in that order. Put every failure in exactly one bucket and record it in STATE › findings as `<job>/<trial> L<line> — <bucket> — <one-line mechanism>`:
   - `infra`: Docker, image build, Harbor, network policy, verifier flakiness, host load.
   - `harness`: rig-coder's prompt, tools, output shaping, loop settings, adapter.
   - `rig`: a defect or limitation in Rig: mis-serialised tool arguments, dropped or mis-assembled stream chunks, unparsed provider fields, wrong error kinds, retry or max-turn handling, effect-log or replay faults, panics. Confirm it against `agent/effects.json` before believing the transcript.
   - `model`: the model could not do it, with a working harness and runtime.
   Also record the three most expensive passes and what they spent it on.
5. **Fix**, most frequent `harness` or `rig` mechanism first, one change per commit (see the next section). `infra` and `model` are logged, never fix targets, unless one mechanism exceeds a third of the rung's failures, in which case stop and report.
6. **Report** in STATE › rungs: pass count, mean, Wilson interval, cost, cost per resolved trial, bucket counts, the commits kept with per-task deltas, and two or three observations. Update the PR description's scoreboard (`gh pr edit`). Commit and push, then advance `current_rung`.

## Making a change

- **Agent changes** live in `test-support/rig-coder/`. They must be task-agnostic: a rule in `prompt.md` states a general working habit, never a fact about one task.
- **Rig changes** live in `crates/`. Write a minimal failing Rust test in the owning crate that reproduces the defect without a network (a cassette, a mock model, or a unit test), then fix it, then see the test pass. Follow `AGENTS.md` for where code and tests go; breaking API changes are allowed when the result is better, and the PR description's `## Changelog` and `## Migration` sections record them (never edit `CHANGELOG.md` or `MIGRATING.md`). Run the owning crate's tests (`cargo test -p <crate>`) before committing a Rig change; that is the one extra check, because other Rig users depend on it.
- **Keep rule.** Re-run the tasks whose failure the change targets, plus a regression sample of at least five tasks from the same rung that passed before (more if the change touches the prompt or a tool's output shape), with `-i <task>` per task, in one job. Compare with `bench/ledger.py compare <before job> <after job>`. Keep the change only if the targeted mechanism is gone in the re-run, no previously passing task in the sample fails for a `harness` or `rig` reason, and cost per resolved trial did not rise by more than 25%. Correctness fixes in Rig with a passing regression test are kept even when the benchmark does not move. Otherwise revert and record the attempt in STATE › notes.
- **Commit message.** Conventional Commit form: `fix(rig-coder): …`, `feat(rig-coder): …`, `fix(rig-agent): …`. The body names the failure mechanism, the trials it came from, the per-task before and after, and cost per resolved trial before and after. Push after every commit.
- **Keeping up with `main`.** Between rungs, `git fetch origin main` and rebase if `main` moved; fix what breaks, rebuild, re-run the install check, and `git push --force-with-lease`. A rebase is never mixed with a fix.

## On (re)start

1. `git status`, `git log --oneline -5`, `gh pr view <STATE › pr> --json state,statusCheckRollup,comments` (address maintainer review comments on the PR first).
2. If STATE › in_progress_job is set: if `runs/<job>/result.json` exists and the ledger has no row for it, record it; if the job is incomplete, record the finished trials and reserve the unfinished ones at the rung's highest observed trial cost.
3. Check that the receipt in `bench/bin/` matches HEAD for the agent's inputs; rebuild if not.
4. Resume at `current_rung`, step `current_step`.

## Stop conditions

Stop and write a report in STATE › notes, then commit and push, when: the next job would cross the budget; the same `harness` or `rig` mechanism survives two fixes; more than a third of a rung's failures are `infra`; the egress probe reaches any URL; a task's result looks contaminated and the allowlist cannot explain how; or a Rig fix would need a change to the effect-log wire format or the provider contract that you cannot test offline. The report gives bucket counts per rung, the kept commits with their deltas, the cost per resolved trial trend, the spend, and what the next start should do.

---

# STATE (edit below this line only)

```yaml
version: 1
pr: null                   # PR URL once opened
budget_usd_cap: 400
spent_usd_known: 0.00
spent_usd_reserved_unknown: 0.00
current_rung: 1
current_step: prepare      # prepare | run | ledger | analyse | fix | report
in_progress_job: null      # runs/<job> while a job is live
last_started: null         # ISO date of the last (re)start
```

## rungs
<!-- one entry per finished rung:
### <n> <dataset@version> (<date>)
job: runs/<job>  passed: x/y  mean: 0.xx  wilson: [a, b]  cost: $x  per-resolved: $x
buckets: infra n · harness n · rig n · model n · contaminated n
kept: <commit> <one line> (tasks: before→after)
observations: … -->

## findings
<!-- every failed or errored trial, once: <job>/<trial> L<line> — <bucket> — <mechanism>; and cost outliers -->

## notes
<!-- lessons for the next start: adapter quirks, dataset quirks, reverted attempts, what wasted money -->
