# Rig-Gemini-gRPC

This companion crate integrates Google Gemini gRPC API with Rig, offering better performance and type safety compared to the REST API.

## Installation

Add the companion crate to your `Cargo.toml`, along with rig-core:

```toml
[dependencies]
rig-gemini-grpc = "0.1.0"
rig-core = "0.28.0"
```

## Setup

Set your Gemini API key as an environment variable:

```shell
export GEMINI_API_KEY=your_api_key_here
```

## Example

```rust
use rig::prelude::*;
use rig_gemini_grpc::Client;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = Client::from_env();

    let agent = client
        .agent("gemini-2.5-flash")
        .preamble("You are a helpful assistant.")
        .build();

    let response = agent.prompt("Hello!").await?;
    println!("{}", response);

    Ok(())
}
```

## Features

- Full completion support with streaming
- Embedding generation
- Tool calling support
- Reasoning and thought signatures
- Image input support

## Migrating from rig-core's gemini_grpc

If you were using the gemini_grpc provider from rig-core, update your code:

**Before** (rig-core 0.28.0):
```rust
use rig::providers::gemini_grpc;

let client = gemini_grpc::Client::from_env();
let model = client.completion_model(gemini_grpc::completion::GEMINI_2_5_FLASH);
```

**After** (rig-gemini-grpc 0.1.0):
```toml
# Add to Cargo.toml
[dependencies]
rig-gemini-grpc = "0.1.0"
```

```rust
use rig_gemini_grpc::{Client, completion::GEMINI_2_5_FLASH};

let client = Client::from_env();
let model = client.completion_model(GEMINI_2_5_FLASH);
```

## More Examples

See the [`/examples`](./examples) folder for more usage examples.
