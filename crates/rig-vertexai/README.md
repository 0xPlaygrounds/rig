## Rig-VertexAI

This companion crate integrates Google Cloud Vertex AI as a model provider with Rig: Gemini through `GenerateContent`, and Anthropic's Claude through the `anthropic` publisher's `rawPredict` endpoint.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-vertexai = "0.46.0"
rig-core = "0.46.0"
```

You can also run `cargo add rig-vertexai rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.
## Raw responses

A response's `raw` is the reply's REST JSON: the SDK's `GenerateContentResponse`
serialized, with enum values spelled by name. It can be recovered from a
normalized response without another RPC:

```rust
use google_cloud_aiplatform_v1::model::GenerateContentResponse;
use rig_core::{completion::CompletionResponse, serde_json};

fn recover(response: CompletionResponse) -> Result<GenerateContentResponse, serde_json::Error> {
    serde_json::from_value(response.raw)
}
```

## Claude

`VertexAi::anthropic(model)` sends Anthropic's Messages API to Vertex AI's
`rawPredict` endpoint (`streamRawPredict` when streamed) with the client's
Google credentials. Requests, replies, Anthropic typed options and history
rules are Anthropic's; the model is named in the URL and the body carries
`anthropic_version: "vertex-2023-10-16"`. Use Vertex AI's model ids, such as
`claude-sonnet-4-6` or `claude-haiku-4-5@20251001`. `global` and the `us` and
`eu` multi-regions have their own hosts; any other location is a region.

```rust,no_run
use rig_core::completion::CompletionRequest;
use rig_vertexai::VertexAi;

# async fn example() -> anyhow::Result<()> {
let claude = VertexAi::from_env()?.anthropic("claude-sonnet-4-6");
let reply = claude.call(CompletionRequest::new("Hello, Claude").max_tokens(256)).await?;
# let _ = reply;
# Ok(())
# }
```

Claude requests travel over HTTP, through the shared reqwest client unless
`VertexAiBuilder::with_http_client` supplies another. They need the
credentials the client resolves, so a client built on a supplied prediction
service cannot send them.

## Setup

Make sure to have Google Cloud credentials configured. You can use Application Default Credentials (ADC) by running:

```shell
gcloud auth application-default login
```

## Supplying your own SDK client

`VertexAi::from_env()` (and `VertexAi::builder()` without either explicit credentials
or a supplied service) resolves Application Default Credentials. That resolution builds
`google-cloud-auth`'s token cache, which **spawns a refresh task on the current
Tokio runtime while it is being constructed**: call it inside a runtime
context, and keep that runtime alive and driven for as long as the client is
used. The refresh task belongs to the runtime, not to any one completion.

A host that would rather own the connection and its credential lifetime can
build the Google SDK client itself and hand it over. Rig then takes the
endpoint, credentials, transport, retry policy and universe domain as given,
and never rebuilds the client or reads ADC:

```rust,no_run
use google_cloud_aiplatform_v1::client::PredictionService;

# async fn example() -> anyhow::Result<()> {
let service = PredictionService::builder()
    .with_endpoint("https://us-central1-aiplatform.googleapis.com")
    .build()
    .await?;

let client = rig_vertexai::VertexAi::builder()
    .with_project("my-project")
    .with_location("us-central1")
    .with_prediction_service(service)
    .build()?;
# let _ = client;
# Ok(())
# }
```

`project` and `location` are still configured on the Rig client: they name the
model resource in the request, not the connection. Supplying both a
prediction service and `with_credentials(...)` is rejected at build time —
the supplied client's credentials are already fixed.
