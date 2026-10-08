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

✨ If you would like to help spread the word about Rig, please consider starring the repo!

> [!WARNING]
> Here be dragons! As we plan to ship a torrent of features in the following months, future updates **will** contain **breaking changes**. With Rig evolving, we'll annotate changes and highlight migration paths as we encounter them.

## Table of contents

- [Table of contents](#table-of-contents)
- [What is Rig?](#what-is-rig)
- [Features](#features)
- [Runtime choices](#runtime-choices)
- [Who's using Rig?](#who-is-using-rig)
- [Get Started](#get-started)
  - [Simple example](#simple-example)
- [The rig coding agent](#the-rig-coding-agent)
- [Integrations](#supported-integrations)

## What is Rig?
Rig is a Rust library for building scalable, modular, and ergonomic **LLM-powered** applications.

More information about this crate can be found in the [official](https://rig.rs/docs) and [crate](https://docs.rs/rig/latest/rig/) API reference documentation.

## Features
- Agentic workflows that can handle multi-turn streaming and prompting
- A classic agent runtime enabled by default
- Full [GenAI Semantic Convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/) compatibility
- 20+ model providers, all under one singular unified interface
- 10+ vector store integrations, all under one singular unified interface
- Full support for LLM completion and embedding workflows
- Support for transcription, audio generation and image generation model capabilities
- Integrate LLMs in your app with minimal boilerplate
- Browser-WASM (`wasm32-unknown-unknown`) support for the portable core and
  classic runtime — see [target support](crates/rig-agent/README.md#target-support)
  for the full matrix (WASI is not supported; `rig-rmcp`/MCP is native-only)

## Runtime choices

Rig separates portable provider/backend contracts from agent orchestration:

- `rig-core` contains provider-neutral messages, completion models, portable and
  contextual tool contracts, memory and vector-store contracts, and built-in
  provider mappings.
- `rig-agent` contains the classic builder, prompt/streaming traits, typed hooks,
  the live tool registry, extraction, and the serializable `AgentRun` state machine. It
  remains enabled by default.

The root `rig` facade re-exports both at their familiar paths, so most code
depends only on `rig`.

Hosts construct HTTP or SDK models with their chosen authentication, transport
policy and runtime lifetime; the agent runtime executes the resulting
`Model` through one shared adapter. Effect replay uses recorded handlers and
does not require live provider construction.

## Who is using Rig?
Below is a non-exhaustive list of companies and people who are using Rig:
- [St Jude](https://www.stjude.org/) - Using Rig for a chatbot utility as part of [`proteinpaint`](https://github.com/stjude/proteinpaint), a genomics visualisation tool.
- [Coral Protocol](https://www.coralprotocol.org/) - Using Rig extensively, both internally as well as part of the [Coral Rust SDK.](https://github.com/Coral-Protocol/coral-rs)
- [VT Code](https://github.com/vinhnx/vtcode) - VT Code is a Rust-based terminal coding agent with semantic code intelligence via Tree-sitter and ast-grep. VT Code uses `rig` for simplifying LLM calls and implementing the model picker.
- [Con](https://github.com/nowledge-co/con) - Con is a GPU-accelerated terminal emulator with a built-in AI agent harness. It uses Rig as the provider abstraction layer for its integrated coding agents.
- [Dria](https://dria.co/) - a decentralised AI network. Currently using Rig as part of their [compute node.](https://github.com/firstbatchxyz/dkn-compute-node)
- [Nethermind](https://www.nethermind.io/) - Using Rig as part of their [Neural Interconnected Nodes Engine](https://github.com/NethermindEth/nine) framework.
- [Neon](https://neon.com) - Using Rig for their [app.build](https://github.com/neondatabase/appdotbuild-agent) V2 reboot in Rust.
- [Listen](https://github.com/piotrostr/listen) - A framework aiming to become the go-to framework for AI portfolio management agents. Powers [the Listen app.](https://app.listen-rs.com/)
- [Cairnify](https://cairnify.com/) - helps users find documents, links, and information instantly through an intelligent search bar. Rig provides the agentic foundation behind Cairnify’s AI search experience, enabling tool-calling, reasoning, and retrieval workflows.
- [Ryzome](https://ryzome.ai) - Ryzome is a visual AI workspace that lets you build interconnected canvases of thoughts, research, and AI agents to orchestrate complex knowledge work.
- [deepwiki-rs](https://github.com/sopaco/deepwiki-rs) - Turn code into clarity. Generate accurate technical docs and AI-ready context in minutes—perfectly structured for human teams and intelligent agents.
- [Cortex Memory](https://github.com/sopaco/cortex-mem) - The production-ready memory system for intelligent agents. A complete solution for memory management, from extraction and vector search to automated optimization, with a REST API, MCP, CLI, and insights dashboard out-of-the-box.
- [Ironclaw](https://github.com/nearai/ironclaw) - A secure personal AI assistant
- [ilert](https://www.ilert.com/) - Incident management & alerting platform. Uses Rig as the multi-provider abstraction in its agentic LLM proxy powering ilert AI.
- [Archestra](https://github.com/archestra-ai/archestra) - MCP-native secure AI platform. Uses Rig in its agentic benchmark.

For a curated list of Rig projects, libraries, tools, articles, and production users, check out [awesome-rig](https://github.com/0xPlaygrounds/awesome-rig).

Are you also using Rig? [Open an issue](https://www.github.com/0xPlaygrounds/rig/issues) to have your name added!

## Get Started
Use the root `rig` facade when you want feature-gated access to companion crates,
or use `rig-core` directly when you only need the core provider abstractions.

```bash
cargo add rig
# or: cargo add rig-core
```

### Simple example
```rust
use rig::prelude::*;
use rig::providers::openai::{self, OpenAI};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // The client reads `OPENAI_API_KEY` and builds the models it serves.
    let model = OpenAI::from_env()?.completion(openai::GPT_5_2);
    let comedian_agent = AgentBuilder::new(model)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;

    println!("{}", response.output());

    Ok(())
}
```
Note using `#[tokio::main]` requires you enable tokio's `macros` and `rt-multi-thread` features
or just `full` to enable all features (`cargo add tokio --features macros,rt-multi-thread`).

More examples live in [`examples`](./examples) and each crate's `examples` directory; provider test coverage and cassette commands are described in [`tests/README.md`](./tests/README.md). Detailed walkthroughs are published on our [Dev.to Blog](https://dev.to/0thtachi) and at [rig.rs/docs](https://rig.rs/docs).

## Recording and replay

With the `cassette` feature, `rig::cassette::effect_log` provides logs,
recorders, replay handlers and checkpoints. Keep an `EffectLogRecorder` handle
and attach its clone with `AgentBuilder::record_to`; import
`rig::cassette::agent::AgentReplayExt` to stamp the resulting log or check
replay compatibility.

For transport-free consumers, depend directly on `rig-cassette` with default
features disabled. Its optional `agent` adapter is independent of the native
`http` engine. The agent runtime does not depend on the concrete logging
crate. See the [cassette README](crates/rig-cassette/README.md) for
dependency guarantees and migration paths.

## The rig coding agent

`cargo install rig` also installs `rig`, a terminal coding agent. The agent is
the [`rig-code`](crates/rig-code) crate, a Bevy app, and `rig` is its launcher:
it generates a small Cargo project for the agent, builds it, and runs it.

```bash
cargo install rig
export OPENAI_API_KEY=...   # or any other provider key in the model catalog
rig                         # builds the agent (later starts rebuild what changed), then opens it
```

In the agent, `/model` picks a model (providers that need no key, such as a
local Ollama, are listed last), `/effort` its reasoning setting, `/help`
lists the commands, and Esc stops a running turn. The status line shows the
session's tokens (uncached input, output, cache reads and writes), its cost
(the provider's figure, else the catalog's list price; `+` when some calls had
no price) and the context in use against the model's window; `/usage` breaks
them down. `/reload` rebuilds the agent
and restarts it on the same session; it shows cargo's progress, keeps the
current build running if the new one does not compile (Esc closes the
compiler output it shows), and rolls back to it if
the new one crashes during startup. The session is saved after every turn;
if the agent crashes or the terminal closes, the next `rig` in the same
directory resumes it. `/quit` ends it.

The system prompt includes the instruction files `AGENTS.md` (or `CLAUDE.md`)
of `RIG_HOME`, of the working directory and of each directory above it, from
the most general to the most specific, at most 32 KB each and 64 KB together,
along with the working directory, platform, date and git branch. They are
re-read when a turn starts, so an edited `AGENTS.md` counts from the next
message; `/context` re-reads them now and lists what the prompt holds.

Every file lives under `RIG_HOME` (default `~/.rig`): the plugin list
`plugins.toml`, the generated `project/`, cargo's `target/`, the builds in `bin/`,
and `sessions/<id>/` with the saved state, the effect log `effects.jsonl` and
the log `agent.log`. `target/` holds cargo's build of the agent and takes a few
gigabytes; set `RIG_HOME` to put everything elsewhere, for example under a
cache directory. Several `rig` processes can share one `RIG_HOME`.

To run the agent from a rig checkout instead of crates.io, install the
launcher from it or point `RIG_SOURCE` at it:

```bash
cd rig && cargo install --path . --root /some/dir
RIG_HOME=/some/dir/home RIG_SOURCE=$PWD /some/dir/bin/rig
```

Plugins are Bevy plugins. A plugin crate depends on `rig-code` and on Bevy
crates at exactly `=0.20.0-rc.2`, and registers tools and slash commands the
same way the built-in ones are registered:

```rust,ignore
use rig_code::prelude::*;

#[derive(Default)]
pub struct HelloPlugin;

impl Plugin for HelloPlugin {
    fn build(&self, app: &mut App) {
        app.add_command("hello", "Say hello", hello);
        // app.add_tool(MyTool) adds any rig_core::tool::Tool. A tool that
        // blocks wraps that work in `blocking(|| ...)`.
    }
}

fn hello(In(args): In<CommandArgs>, mut notices: MessageWriter<Notice>) {
    notices.write(Notice::info(args.agent, "Hello!"));
}
```

List it in `$RIG_HOME/plugins.toml` and run `/reload`. The built-in tools,
the built-in commands and the terminal view are entries in the same list, so
any of them can be removed or replaced:

```toml
[[plugin]]
plugin = "rig_code::builtin::BuiltinToolsPlugin"

[[plugin]]
plugin = "rig_code::builtin::BuiltinCommandsPlugin"

[[plugin]]
plugin = "rig_code::tui::TuiPlugin"

[[plugin]]
crate = "rig-hello"               # the package name
path = "../rig-hello"             # relative to plugins.toml; or git = "..." (branch, rev), or version = "..."
plugin = "rig_hello::HelloPlugin" # implements Plugin + Default
bevy_features = []                # optional extra Bevy features
```

The agent, like the rest of the workspace, needs Rust 1.96 or newer.

`rig build` regenerates and builds the agent without starting it.
`CARGO_BUILD_JOBS` sets cargo's `-j` for these builds, as for any cargo build.

The agent runs on Linux and macOS; Windows is not supported yet.

## Supported Integrations

The built-in `rig::vector_store::in_memory_store::InMemoryVectorStore` stores
serializable documents without requiring `Eq` or `Default`. Its custom ID
callbacks accept closures that capture and mutate application state. See the
[vector-search example](examples/vector_search/src/main.rs).

The root `rig` facade exposes companion crates behind one feature per integration:

```toml
rig = { version = "0.36.0", features = ["lancedb", "fastembed"] }
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
| Memory policies | [`rig-memory`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-memory) | `memory` | `rig::memory` |
| Milvus | [`rig-milvus`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-milvus) | `milvus` | `rig::milvus` |
| MongoDB | [`rig-mongodb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-mongodb) | `mongodb` | `rig::mongodb` |
| Neo4j | [`rig-neo4j`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-neo4j) | `neo4j` | `rig::neo4j` |
| PostgreSQL | [`rig-postgres`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-postgres) | `postgres` | `rig::postgres` |
| Qdrant | [`rig-qdrant`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-qdrant) | `qdrant` | `rig::qdrant` |
| ScyllaDB | [`rig-scylladb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-scylladb) | `scylladb` | `rig::scylladb` |
| SQLite | [`rig-sqlite`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-sqlite) | `sqlite` | `rig::sqlite` |
| SurrealDB | [`rig-surrealdb`](https://github.com/0xPlaygrounds/rig/tree/main/crates/rig-surrealdb) | `surrealdb` | `rig::surrealdb` |
| TypeSafe Jev (experimental judgments) | [`rig-typesafeai`](crates/rig-typesafeai) | `typesafeai` | `rig::typesafeai` |

`rig::memory` is available without the `memory` feature; it contains the core
conversation memory traits and in-memory backend re-exported from `rig-core`.
Enabling `features = ["memory"]` adds reusable history-shaping policy types from
the `rig-memory` companion crate to the same module.

We also have some other associated crates that have additional functionality you may find helpful when using Rig:
- `rig-onchain-kit` - the [Rig Onchain Kit.](https://github.com/0xPlaygrounds/rig-onchain-kit) Intended to make interactions between Solana/EVM and Rig much easier to implement.


<p align="center">
<br>
<br>
<img src="img/built-by-playgrounds.svg" alt="Build by Playgrounds" width="30%">
</p>
