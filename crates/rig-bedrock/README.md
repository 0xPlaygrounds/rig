## Rig-Bedrock
This companion crate integrates AWS Bedrock as a model provider with Rig.

It supports two independent paths:

| Path | Module | Auth | Models |
|------|--------|------|--------|
| **Converse** | `client`, `completion`, … | AWS SDK credential chain | Foundation models via Converse / ConverseStream |
| **Mantle** | `mantle` | Short-term IAM bearer token or `AWS_BEARER_TOKEN_BEDROCK` | OpenAI-compatible models (e.g. GPT-OSS) |

Converse and Mantle do not auto-route model IDs between each other.

## Usage

Add the companion crate to your `Cargo.toml`, along with the rig-core crate:

```toml
[dependencies]
rig-bedrock = "0.40.0"
rig-core = "0.40.0"
```

You can also run `cargo add rig-bedrock rig-core` to add the most recent versions of the dependencies to your project.

See the [`/examples`](./examples) folder for usage examples.

### Converse (classic Bedrock Runtime)

Make sure to have AWS credentials env vars loaded before starting the client:

```shell
export AWS_DEFAULT_REGION=us-east-1
export AWS_SECRET_ACCESS_KEY=.......
export AWS_ACCESS_KEY_ID=......
```

```rust,ignore
use rig_bedrock::{client::Client, completion::AMAZON_NOVA_LITE};
use rig_core::client::{CompletionClient, ProviderClient};

let client = Client::from_env()?;
let agent = client.agent(AMAZON_NOVA_LITE).build();
```

### Mantle (OpenAI-compatible)

Mantle serves selected models through `https://bedrock-mantle.{region}.api.aws/openai/v1`.
Auth is either:

1. **Short-term IAM token** — minted via SigV4 (`Action=CallWithBearerToken`, 12h TTL), or
2. **`AWS_BEARER_TOKEN_BEDROCK`** — a pre-minted `bedrock-api-key-…` value (skips minting).

```shell
export AWS_REGION=us-east-1
# optional: export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...
# otherwise uses the default AWS credential chain to mint a token
```

```rust,ignore
use rig_bedrock::mantle::{ClientBuilder, OPENAI_GPT_OSS_20B};
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

let client = ClientBuilder::from_env().await?;
let agent = client.agent(OPENAI_GPT_OSS_20B).build();
let reply = agent.prompt("Hello").await?;
```

Useful model constants:

- `OPENAI_GPT_OSS_20B` / `OPENAI_GPT_OSS_120B` — versioned ids (`openai.gpt-oss-20b-1:0`, …)
- `OPENAI_GPT_OSS_20B_MANTLE` / `OPENAI_GPT_OSS_120B_MANTLE` — unversioned aliases used in some AWS examples

Example binary: `cargo run -p rig-bedrock --example agent_with_bedrock_mantle`.
