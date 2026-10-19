<p align="center">
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="img/rig-rebranded-logo-white.svg">
    <source media="(prefers-color-scheme: light)" srcset="img/rig-rebranded-logo-black.svg">
    <img src="img/rig-rebranded-logo-white.svg" style="width: 40%; height: 40%;" alt="Rig logo">
</picture>
<br>
<br>
<a href="https://rig.rs/docs"><img src="https://img.shields.io/badge/📖 docs-rig.rs-dca282.svg" /></a> &nbsp;
<a href="https://docs.rs/rig/latest/rig/"><img src="https://img.shields.io/badge/docs-API Reference-dca282.svg" /></a> &nbsp;
<a href="https://crates.io/crates/rig"><img src="https://img.shields.io/crates/v/rig.svg?color=dca282" /></a>
&nbsp;
<a href="https://crates.io/crates/rig"><img src="https://img.shields.io/crates/d/rig-core.svg?color=dca282" /></a>
&nbsp;
<a href="LICENSE"><img src="https://img.shields.io/crates/l/rig.svg?color=dca282" /></a>
</br>
<a href="https://discord.gg/playgrounds"><img src="https://img.shields.io/discord/511303648119226382?color=%236d82cc&label=Discord&logo=discord&logoColor=white" /></a>
&nbsp;
<a href=""><img src="https://img.shields.io/badge/built_with-Rust-dca282.svg?logo=rust" /></a>
&nbsp;
<a href="https://github.com/0xPlaygrounds/rig"><img src="https://img.shields.io/github/stars/0xPlaygrounds/rig?style=social" alt="stars - rig" /></a>
<br>

<br>
</p>
&nbsp;


<div align="center">

[📑 Docs](https://rig.rs/docs)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[🌐 Website](https://rig.rs)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[🤝 Contribute](https://github.com/0xPlaygrounds/rig/issues/new)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
[✍🏽 Blogs](https://rig.rs/docs/guides)
<span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
<a href="https://ryzome.ai"><img src="img/ryzome-bg.png" height="32" align="absmiddle" alt="Ryzome" /></a>

</div>

Rig is a Rust library for building LLM applications, and `rig`, a terminal
coding agent built on it that you extend with Bevy plugins.

## The coding agent

```bash
cargo install rig
rig
```

In the agent, `/login chatgpt` signs in with a ChatGPT plan, or export a
provider's API key (such as `OPENAI_API_KEY`) before starting. `/model` picks
the model and `/help` lists the commands. `rig -p "fix the failing test"`
answers one prompt without the terminal view.

- **Everything is a plugin.** The agent is a [Bevy](https://bevy.org) app; its
  tools, commands, sessions, sign-in and terminal view are each a plugin listed
  in `plugins.toml`, so any of them can be removed, replaced or added to.
- **It rebuilds itself.** `/reload` rebuilds the agent with your plugin
  changes and restarts it in the same session, and keeps the old build running
  if the new one does not compile.
- **Scriptable.** `rig -p --json` and `rig --json` drive it over pipes;
  `rig help-agents` is the protocol.
- **More when you want it.** Subagents and checkpoints of what `shell`
  changed are optional plugins, and the tools work on remote hosts over ssh
  or any command.

The [agent guide](docs/agent.md) covers the keys, sessions, compaction,
plugins and files, and the [plugin guide](crates/rig-harness/PLUGINS.md) shows
how to write a tool, command or panel. The agent needs Rust 1.97.1 or newer
and runs on Linux and macOS.

## The library

```bash
cargo add rig
```

```rust
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // The client reads `OPENAI_API_KEY` and builds the models it serves.
    let model = OpenAI::from_env()?.completion(openai::GPT_6_1_SOL);
    let comedian = AgentBuilder::new(model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    let response = comedian.prompt("Entertain me!").await?;
    println!("{}", response.output());
    Ok(())
}
```

`#[tokio::main]` needs tokio's `macros` and `rt-multi-thread` features
(`cargo add tokio --features macros,rt-multi-thread`).

- 20+ model providers and 10+ vector stores, each under one interface
- Agents with tools, multi-turn streaming and typed hooks
- Completion, embedding, transcription, audio and image generation
- [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) for OpenTelemetry
- Recording and replay of an agent's effects ([`rig-cassette`](crates/rig-cassette/README.md))
- Browser WASM (`wasm32-unknown-unknown`) for the core and the agent runtime ([target support](crates/rig-agent/README.md#target-support))

More in [`examples/`](examples), the [docs](https://rig.rs/docs) and the
[API reference](https://docs.rs/rig/latest/rig/).

## Crates

The `rig` crate re-exports the core and the agent runtime, and each companion
crate behind a feature.

| Crate | What it is |
| --- | --- |
| [`rig-core`](crates/rig-core) | Provider-neutral messages, completion and embedding models, tool contracts, the conversation type, vector-store contracts, and the providers |
| [`rig-agent`](crates/rig-agent) | The agent builder, prompting and streaming, hooks, the tool registry and the serializable `AgentRun` |
| [`rig-memory`](crates/rig-memory) | Windows, compaction, and saving, resuming and forking conversations |
| [`rig-cassette`](crates/rig-cassette) | Effect logs: recording, replay and checkpoints |
| [`rig-tools`](crates/rig-tools) | Coding tools any rig agent can use: read, write, edit, search, shell |
| [`rig-ecs`](crates/rig-ecs), [`rig-harness`](crates/rig-harness) | The agent kernel on Bevy's ECS, and the harness the coding agent's plugins build on |
| [`rig-tui`](plugins/rig-tui), [`rig-basics`](plugins/rig-basics) | The terminal view, and the coding agent's default plugins |

### Integrations

```toml
rig = { version = "0.44.0", features = ["lancedb", "fastembed"] }
```

| Integration | Crate | Feature | Module path |
| --- | --- | --- | --- |
| AWS Bedrock | [`rig-bedrock`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-bedrock) | `bedrock` | `rig::bedrock` |
| AWS S3Vectors | [`rig-s3vectors`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-s3vectors) | `s3vectors` | `rig::s3vectors` |
| Candle (local Llama/SmolLM2/Qwen3 tools, YOLOv8 pose) | [`rig-candle`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-candle) | `candle` | `rig::candle` |
| Cloudflare Vectorize | [`rig-vectorize`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-vectorize) | `vectorize` | `rig::vectorize` |
| FastEmbed | [`rig-fastembed`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-fastembed) | `fastembed` | `rig::fastembed` |
| Google Gemini gRPC | [`rig-gemini-grpc`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-gemini-grpc) | `gemini-grpc` | `rig::gemini_grpc` |
| Google Vertex AI | [`rig-vertexai`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-vertexai) | `vertexai` | `rig::vertexai` |
| HelixDB | [`rig-helixdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-helixdb) | `helixdb` | `rig::helixdb` |
| LanceDB | [`rig-lancedb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-lancedb) | `lancedb` | `rig::lancedb` |
| Conversations over time | [`rig-memory`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-memory) | `memory` | `rig::memory` |
| Milvus | [`rig-milvus`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-milvus) | `milvus` | `rig::milvus` |
| MongoDB | [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-mongodb) | `mongodb` | `rig::mongodb` |
| Neo4j | [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-neo4j) | `neo4j` | `rig::neo4j` |
| PostgreSQL | [`rig-postgres`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-postgres) | `postgres` | `rig::postgres` |
| Qdrant | [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-qdrant) | `qdrant` | `rig::qdrant` |
| ScyllaDB | [`rig-scylladb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-scylladb) | `scylladb` | `rig::scylladb` |
| SQLite | [`rig-sqlite`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-sqlite) | `sqlite` | `rig::sqlite` |
| SurrealDB | [`rig-surrealdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-surrealdb) | `surrealdb` | `rig::surrealdb` |
| TypeSafe Jev (experimental judgments) | [`rig-typesafeai`](crates/rig-typesafeai) | `typesafeai` | `rig::typesafeai` |

## Community

Projects built on Rig are listed in
[awesome-rig](https://github.com/0xPlaygrounds/awesome-rig); open a pull
request there to add yours. Questions and ideas are welcome on
[Discord](https://discord.gg/playgrounds), and [CONTRIBUTING.md](CONTRIBUTING.md)
says how to contribute. Rig is MIT licensed.

> [!WARNING]
> Rig is pre-1.0. Releases contain breaking changes; each release's
> [changelog](CHANGELOG.md) and [migration notes](MIGRATING.md) say what
> changed and how to move.

<p align="center">
<br>
<br>
<img src="img/built-by-playgrounds.svg" alt="Build by Playgrounds" width="30%">
</p>
