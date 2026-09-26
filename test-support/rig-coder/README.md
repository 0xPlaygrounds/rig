# rig-coder

An unpublished coding agent built on Rig's default agent runtime, used to find
and fix Rig defects by running coding benchmarks through
[Harbor](https://github.com/harbor-framework/harbor). The agent is a system
prompt, six workspace tools (`bash`, `read_file`, `write_file`, `edit_file`,
`list_files`, `grep`) and a hook that writes a live JSONL transcript. Each run
also writes its effect log.

```sh
export GEMINI_API_KEY=...
cargo run -p rig-coder -- --cwd /path/to/repo "Fix the failing test in src/lib.rs"
```

Flags: `--task-file`, `--provider gemini|anthropic|openai`, `--model`,
`--max-turns` (200), `--max-tokens`, `--timeout-secs`, `--transcript`,
`--effect-log`. The exit code is 0 when the run settles, 1 when it fails, 2 on
timeout and 3 on a setup error.

`bench/` holds the Harbor adapter, the Linux build script, the dataset egress
policy and the ledger. [LADDER.md](LADDER.md) is the restartable prompt that
drives the benchmark ladder and records its state.
