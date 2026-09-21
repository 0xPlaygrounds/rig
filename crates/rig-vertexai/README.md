## Rig-VertexAI

This companion crate integrates Google Cloud Vertex AI (hosted models including Gemini) as a model provider with Rig.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-vertexai = "0.42.0"
rig-core = "0.42.0"
```

You can also run `cargo add rig-vertexai rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

## Raw responses

`CompletionModel::raw_completion` returns the public
`rig_vertexai::completion::VertexGenerateContentOutput` wrapper. It can be stored
in typed library APIs or recovered from a normalized response without another RPC:

```rust
use rig_core::{completion::CompletionResponse, serde_json};
use rig_vertexai::completion::VertexGenerateContentOutput;

fn recover(response: CompletionResponse) -> Result<VertexGenerateContentOutput, serde_json::Error> {
    serde_json::from_value(response.raw)
}
```

With the facade's `vertexai` feature, the same type is available at
`rig::vertexai::completion::VertexGenerateContentOutput`.

## Setup

Make sure to have Google Cloud credentials configured. You can use Application Default Credentials (ADC) by running:

```shell
gcloud auth application-default login
```
