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

## Concrete client and data APIs

`rig_bedrock::Client` is a concrete, monomorphic connection handle. It owns
reusable AWS connection settings (and can retain a caller-built SDK client)
while materializing plain `Config`, `EmbeddingConfig`, and `ImageConfig`
records. The low-level `rig_bedrock::functions` API remains public and takes
the live AWS SDK client explicitly.

```rust,ignore
use rig_bedrock::{Client, completion::AMAZON_NOVA_LITE, functions};
use rig_core::completion::CompletionRequest;

let client = Client::from_env();
let cfg = client.config(AMAZON_NOVA_LITE);
let aws = client.get_inner().await;

let response = functions::complete(
    &aws,
    &cfg.model,
    CompletionRequest::from_prompt("Describe the solar system"),
)
.await?;
```

Streaming is `functions::open_stream`, embeddings are `functions::embed` /
`functions::embed_batches`, and image generation is
`functions::generate_image`.

With `rig-agent`'s `bedrock` feature, the same client provides the fluent agent
bridge through the prelude:

```rust,ignore
use rig_agent::prelude::*;
use rig_bedrock::{Client, completion::AMAZON_NOVA_LITE};

let client = Client::from_env();
let agent = client
    .agent(AMAZON_NOVA_LITE)
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
