# Rig-Gemini-gRPC

This companion crate integrates Google Gemini gRPC API with Rig, offering better performance and type safety compared to the REST API.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-gemini-grpc = "0.2.5"
rig-core = "0.36.0"
```

You can also run `cargo add rig-gemini-grpc rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for more usage examples.

## Setup

Set your Gemini API key as an environment variable:

```shell
export GEMINI_API_KEY=your_api_key_here
```

## Entry point: `functions`

The crate's face is a serde `functions::Config` plus free functions. Because
gRPC is a non-HTTP transport, the connected tonic channel cannot be plain
data: `functions::client_from_config` turns a config into the live `Client`
handle, which every free function takes.

```rust
use rig_core::completion::CompletionRequest;
use rig_gemini_grpc::functions;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Reads GEMINI_API_KEY from the environment.
    let cfg = functions::Config::new("gemini-2.5-flash");
    let client = functions::client_from_config(&cfg).await?;

    let response = functions::complete(
        &client,
        &cfg.model,
        CompletionRequest::from_prompt("Hello!"),
    )
    .await?;
    println!("{:?}", response.choice);

    Ok(())
}
```

## As an agent

The same config drops into `rig-agent` (feature `gemini-grpc`), which builds
and caches the channel for you:

```rust
use rig_agent::{agent::AgentBuilder, provider::ProviderConfig};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let agent = AgentBuilder::new(ProviderConfig::GeminiGrpc(
        rig_gemini_grpc::functions::Config::new("gemini-2.5-flash"),
    ))
    .preamble("You are a helpful assistant.")
    .build();

    println!("{}", agent.prompt("Hello!").await?);

    Ok(())
}
```

## Embeddings

`functions::EmbeddingConfig` is the embeddings sibling; pass a client plus
`cfg.model` / `cfg.ndims` to `functions::embed` or `functions::embed_batches`.

## Features

- Full completion support with streaming
- Embedding generation
- Tool calling support
- Reasoning and thought signatures
- Image input support
