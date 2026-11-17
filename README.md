<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/rig-rebranded-logo-white.svg">
  <img src="img/rig-rebranded-logo-black.svg" width="40%" alt="Rig">
</picture>
</p>

[![crates.io](https://img.shields.io/crates/v/rig.svg)](https://crates.io/crates/rig)
[![docs.rs](https://img.shields.io/docsrs/rig)](https://docs.rs/rig)
[![Discord](https://img.shields.io/discord/511303648119226382?label=Discord)](https://discord.gg/playgrounds)

Rig is a Rust library for building applications on large language models.
It also ships `rig`, a terminal coding agent built on the library that you
extend with Bevy plugins.

## The coding agent

Install the launcher, `rig-cli`, from a checkout. The
[agent guide](docs/agent.md#install-and-run) covers installing a release.
It needs Rust 1.97.1 or newer and runs on Linux and macOS.

```bash
git clone https://github.com/0xPlaygrounds/rig && cd rig
cargo install --locked --force --path crates/rig-cli
rig
```

The first `rig` builds the agent, which takes several minutes, and opens it.
Run `/login` to sign in or save an API key, or export one such as
`OPENAI_API_KEY` before starting. `/model` picks a model and `/help` lists
the commands. `rig -p "fix the failing test"` answers one prompt and exits.

Its tools, commands, sessions, sign-in and terminal view are each a plugin,
turned on or off in `~/.rig/settings.toml`. `/reload` rebuilds the agent with
your changes and keeps the session.

- The [agent guide](docs/agent.md) covers updating, models, sessions, tools,
  settings and scripting.
- The [plugin guide](crates/rig-harness/PLUGINS.md) shows how to write a
  tool, a command or a panel.

## The library

```bash
cargo add rig anyhow
cargo add tokio --features macros,rt-multi-thread
```

```rust
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Reads OPENAI_API_KEY.
    let model = OpenAI::from_env()?.completion(openai::GPT_6_1_SOL);
    let agent = AgentBuilder::new(model)
        .preamble("You are a comedian.")
        .build();

    let response = agent.prompt("Entertain me!").await?;
    println!("{}", response.output());
    Ok(())
}
```

Rig puts over 20 model providers and a dozen vector stores behind one
interface. It has agents with tools, streaming and typed hooks. It covers
completion, embedding, transcription, audio and image models and follows the
OpenTelemetry GenAI conventions. `rig-core` and `rig-agent` build for browser
WASM. See [`examples/`](examples), the [guides](https://rig.rs/docs) and the
[API reference](https://docs.rs/rig).

## Crates

The `rig` crate re-exports `rig-core`. Its default features add `rig-agent`,
the derive macros and the reqwest HTTP client with rustls. Each crate in the
list below is a feature named after it without `rig-`:

```bash
cargo add rig --features lancedb,fastembed
```

- Agents and state: [`rig-agent`](crates/rig-agent),
  [`rig-memory`](crates/rig-memory) (saved conversations),
  [`rig-cassette`](crates/rig-cassette) (record and replay),
  [`rig-rmcp`](crates/rig-rmcp) (MCP tools).
- Models: [`rig-bedrock`](crates/rig-bedrock),
  [`rig-vertexai`](crates/rig-vertexai),
  [`rig-gemini-grpc`](crates/rig-gemini-grpc),
  [`rig-candle`](crates/rig-candle) (local models),
  [`rig-fastembed`](crates/rig-fastembed) (local embeddings),
  [`rig-typesafeai`](crates/rig-typesafeai) (TypeSafe Jev judgments).
- Vector stores: [`rig-helixdb`](crates/rig-helixdb),
  [`rig-lancedb`](crates/rig-lancedb), [`rig-milvus`](crates/rig-milvus),
  [`rig-mongodb`](crates/rig-mongodb), [`rig-neo4j`](crates/rig-neo4j),
  [`rig-postgres`](crates/rig-postgres), [`rig-qdrant`](crates/rig-qdrant),
  [`rig-s3vectors`](crates/rig-s3vectors), [`rig-scylladb`](crates/rig-scylladb),
  [`rig-sqlite`](crates/rig-sqlite), [`rig-surrealdb`](crates/rig-surrealdb),
  [`rig-vectorize`](crates/rig-vectorize).

Internals: [`rig-core`](crates/rig-core) (messages, models, tools,
providers), [`rig-derive`](crates/rig-derive) (feature `derive`),
[`rig-http`](crates/rig-http) (transport traits),
[`rig-reqwest`](crates/rig-reqwest) (feature `reqwest`) and
[`rig-tungstenite`](crates/rig-tungstenite) (feature `websocket`).

Coding agent (not facade features): [`rig-cli`](crates/rig-cli),
[`rig-harness`](crates/rig-harness),
[`rig-harness-protocol`](crates/rig-harness-protocol),
[`rig-ecs`](crates/rig-ecs), [`rig-tools`](crates/rig-tools),
[`rig-markdown`](crates/rig-markdown), and the plugins
[`rig-basics`](plugins/rig-basics), [`rig-tui`](plugins/rig-tui),
[`rig-fold`](plugins/rig-fold), [`rig-steel`](plugins/rig-steel),
[`rig-mcp-client`](plugins/rig-mcp-client), [`rig-acp`](plugins/rig-acp),
[`rig-inspect`](plugins/rig-inspect), [`rig-traffic`](plugins/rig-traffic)
and [`rig-claude-cloak`](plugins/rig-claude-cloak).

## Community

Ask questions on [Discord](https://discord.gg/playgrounds).
[CONTRIBUTING.md](CONTRIBUTING.md) says how to contribute, and
[awesome-rig](https://github.com/0xPlaygrounds/awesome-rig) lists projects
built on Rig.

Rig is pre-1.0, so releases break APIs. The [changelog](CHANGELOG.md) and
[migration notes](MIGRATING.md) say what changed. Rig is MIT licensed.
