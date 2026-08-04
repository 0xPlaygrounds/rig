## Rig-VertexAI

This companion crate integrates Google Cloud Vertex AI (hosted models including Gemini) as a model provider with Rig.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-vertexai = "0.3.5"
rig-core = "0.36.0"
```

You can also run `cargo add rig-vertexai rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

## Setup

Make sure to have Google Cloud credentials configured. You can use Application Default Credentials (ADC) by running:

```shell
gcloud auth application-default login
export GOOGLE_CLOUD_PROJECT=my-project
```

## Entry point: `functions`

The crate's face is a serde `functions::Config` (project, location, and the
credential *source* — never key material) plus free functions.

Vertex AI authenticates through Google's ADC/OAuth chain, so the authenticated
handle cannot be plain data: `functions::client_from_config` resolves
credentials into a live `Client`, which every free function takes.

```rust
use rig_core::completion::CompletionRequest;
use rig_vertexai::{completion::GEMINI_2_5_FLASH_LITE, functions};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let config = functions::Config::new(GEMINI_2_5_FLASH_LITE);
    let client = functions::client_from_config(&config)?;

    let response = functions::complete(
        &client,
        &config.model,
        CompletionRequest::from_prompt("What is the capital of France?"),
    )
    .await?;
    println!("{:?}", response.choice);

    Ok(())
}
```

`Config::with_project`, `with_location` and `with_impersonated_service_account`
override the environment defaults.

## Agents and streaming

Because of that OAuth handle, Vertex AI has no `rig-agent` `ProviderConfig`
arm. To run a tool-calling agent loop, drive the public `AgentRun` +
`prepare_request` protocol and call `functions::complete` yourself — see
[`examples/tool_vertexai.rs`](./examples/tool_vertexai.rs).

Streaming is not supported by this integration: `functions::open_stream`
always errors.
