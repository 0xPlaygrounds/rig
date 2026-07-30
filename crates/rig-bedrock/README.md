## Rig-Bedrock
This companion crate integrates AWS Bedrock as model provider with Rig.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-bedrock = "0.4.5"
rig-core = "0.36.0"
```

You can also run `cargo add rig-bedrock rig-core` to add the most recent versions of the dependencies to your project.

## The `functions` face

The crate has no client type and no model traits: everything goes through
`rig_bedrock::functions` — a plain-data `Config` (or `EmbeddingConfig` /
`ImageConfig`) describing how to build an AWS client, plus free functions
taking that client explicitly.

```rust,ignore
use rig_bedrock::{completion::AMAZON_NOVA_LITE, functions};
use rig_core::completion::CompletionRequest;

let cfg = functions::Config::new(AMAZON_NOVA_LITE);
let client = functions::client_from_config(&cfg).await;

let response = functions::complete(
    &client,
    &cfg.model,
    CompletionRequest::from_prompt("Describe the solar system"),
)
.await?;
```

Streaming is `functions::open_stream`, embeddings are `functions::embed` /
`functions::embed_batches`, and image generation is
`functions::generate_image`.

For agents, hand the same config to `rig-agent` (feature `bedrock`):

```rust,ignore
use rig_agent::{agent::AgentBuilder, provider::ProviderConfig};

let agent = AgentBuilder::new(ProviderConfig::Bedrock(
    rig_bedrock::functions::Config::new(rig_bedrock::completion::AMAZON_NOVA_LITE),
))
.preamble("Be precise and concise.")
.build();
```

See the [`/examples`](./examples) folder for usage examples.

Make sure to have AWS credentials env vars loaded before starting client such as:
```shell
export AWS_DEFAULT_REGION=us-east-1
export AWS_SECRET_ACCESS_KEY=.......
export AWS_ACCESS_KEY_ID=......
```
