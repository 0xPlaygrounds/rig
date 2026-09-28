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

## Example

```rust
use rig::prelude::*;
use rig_gemini_grpc::{GeminiGrpc, completion::GEMINI_3_8_FLASH};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let transport = GeminiGrpc::from_env()?;

    let agent = AgentBuilder::new(transport.completion(GEMINI_3_8_FLASH))
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Hello!").await?;
    println!("{}", response.output);

    Ok(())
}
```

Requests and replies normalize through `rig_core::providers::gemini::edge`, the
same layer the REST wires use: thought signatures return on the parts that
carried them, tool schemas are sent as written, and parts rig has no type for
(code execution, inline media) are kept as native content.

## Features

- Full completion support with streaming
- Embedding generation
- Tool calling support
- Reasoning and thought signatures
- Image input support
