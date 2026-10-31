# Rig-Gemini-gRPC

This companion crate sends the Google Gemini API's `GenerateContent` and `EmbedContent` calls over gRPC instead of REST. The request is the Gemini REST provider's request transcoded to protobuf, and each reply is transcoded once to its REST JSON and read by the same decoder, so the history rules, typed options and replies match `rig_core::providers::gemini`.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-gemini-grpc = "0.46.0"
rig-core = "0.46.0"
```

You can also run `cargo add rig-gemini-grpc rig-core` to add the most recent versions of the dependencies to your project, or enable the `rig` facade's `gemini-grpc` feature and use `rig::gemini_grpc`.

See the [`/examples`](./examples) folder for usage examples.

## Setup

Set your Gemini API key as an environment variable:

```shell
export GEMINI_API_KEY=your_api_key_here
```

## Example

```rust,no_run
use rig_core::completion::CompletionRequest;
use rig_core::providers::gemini::GeminiConfig;
use rig_gemini_grpc::GeminiGrpc;

# async fn example() -> anyhow::Result<()> {
let transport = GeminiGrpc::connect(&GeminiConfig::from_env()?).await?;
let reply = transport
    .completion("gemini-2.5-flash")
    .call(CompletionRequest::new("Hello!"))
    .await?;
# let _ = reply;
# Ok(())
# }
```

`GeminiGrpc::connect` reads the same `GeminiConfig` as the REST client and
connects to its base URL. `GeminiGrpc::with_channel` sends through a tonic
channel the host built instead.

## Features

- Completion, unary and streamed
- Embedding generation
- Tool calling
- Reasoning and thought signatures
- Image input
- Typed options and reply extras (`extension::GeminiGrpcExt`)
