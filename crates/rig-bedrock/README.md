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

Mantle serves selected models through an OpenAI-compatible HTTP API. There are **two base URL paths**:

| Base URL | Helper | Use for |
|----------|--------|---------|
| `https://bedrock-mantle.{region}.api.aws/v1` | `openai_base_url` (**default**) | GPT-OSS and most Mantle models (Completions + Responses) |
| `https://bedrock-mantle.{region}.api.aws/openai/v1` | `openai_gpt5_base_url` | GPT-5.x family Responses (e.g. `openai.gpt-5.6-luna`) |

`GET …/v1/models` lists unversioned ids (`openai.gpt-oss-20b`, …). Versioned ids such as `openai.gpt-oss-20b-1:0` are for Bedrock Runtime/Converse, not Mantle.

Auth is either:

1. **Short-term IAM token** — minted via SigV4 (`Action=CallWithBearerToken`, **12h TTL**), or
2. **`AWS_BEARER_TOKEN_BEDROCK`** — a pre-minted `bedrock-api-key-…` value (skips minting).

The token is **snapshotted when the client is built**. Effective lifetime is the minimum of 12 hours (`TOKEN_TTL`) and the source AWS credential session (SSO / AssumeRole sessions are often much shorter). Rebuild the client before that effective TTL elapses.

Mantle clients are first-class Rig types (not OpenAI aliases): defaults use Mantle base URLs, GenAI telemetry reports `aws_bedrock`, and only chat completion is advertised as capable.

```shell
export AWS_REGION=us-east-1
# optional: export AWS_BEARER_TOKEN_BEDROCK=bedrock-api-key-...
# otherwise uses the default AWS credential chain to mint a token
```

```rust,ignore
use rig_bedrock::mantle::{self, OPENAI_GPT_OSS_20B};
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

// Responses API (default OpenAI surface) on /v1
let client = mantle::from_env().await?;
let agent = client
    .agent(OPENAI_GPT_OSS_20B)
    // Mantle Responses often requires store:false
    .additional_params(serde_json::json!({"store": false}))
    .build();
let reply = agent.prompt("Hello").await?;
```

Chat Completions on the same default `/v1` base (works well for GPT-OSS):

```rust,ignore
use rig_bedrock::mantle::{self, ClientBuilder, OPENAI_GPT_OSS_20B};
use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

let client = mantle::from_env_completions().await?;
// or: ClientBuilder::from_env().build_completions().await?
let agent = client.agent(OPENAI_GPT_OSS_20B).build();
let reply = agent.prompt("Hello").await?;
```

GPT-5.x Responses on the alternate `/openai/v1` path:

```rust,ignore
use rig_bedrock::mantle::{openai_gpt5_base_url, ClientBuilder, OPENAI_GPT_5_4};
use rig_core::client::CompletionClient;

let client = ClientBuilder::from_env()
    .base_url(openai_gpt5_base_url("us-east-1"))
    .build()
    .await?;
let agent = client
    .agent(OPENAI_GPT_5_4)
    .additional_params(serde_json::json!({"store": false}))
    .build();
```

Useful model constants:

- `OPENAI_GPT_OSS_20B` / `OPENAI_GPT_OSS_120B` — unversioned Mantle ids
- `OPENAI_GPT_5_4` / `OPENAI_GPT_5_5` — GPT-5.x Mantle ids (use `openai_gpt5_base_url`)
- `OPENAI_GPT_5_6_LUNA` / `OPENAI_GPT_5_6_SOL` / `OPENAI_GPT_5_6_TERRA` — GPT-5.6 family (use `openai_gpt5_base_url`)

Versioned Runtime/Converse ids live on `rig_bedrock::completion` (`OPENAI_GPT_OSS_20B_VERSIONED` / `OPENAI_GPT_OSS_120B_VERSIONED`), not under Mantle.

Mantle HTTPS needs a TLS feature (`rustls` default, or `native-tls`).

Free-form model id strings still work for other Mantle catalog entries.

Example binary: `cargo run -p rig-bedrock --example agent_with_bedrock_mantle`.
