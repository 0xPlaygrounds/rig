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
[`ecs_host_model`](examples/ecs_host_model.rs) shows retained-runtime SDK
preparation and per-poll runtime context on ECS workers, with explicit shutdown
ordering. It makes a live, potentially billable call when run; compilation alone
is not an authentication or service test.

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

## Supplying your own SDK client

`Client::from_env()` (and `Client::builder()` without either explicit credentials
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

let client = rig_vertexai::Client::builder()
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
