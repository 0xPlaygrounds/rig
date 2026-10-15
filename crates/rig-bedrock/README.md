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

See the [`/examples`](./examples) folder for usage examples.

Make sure to have AWS credentials env vars loaded before starting client such as:
```shell
export AWS_DEFAULT_REGION=us-east-1
export AWS_SECRET_ACCESS_KEY=.......
export AWS_ACCESS_KEY_ID=......
```

## Anthropic Messages and OpenAI-compatible APIs

`client::BedrockRuntime` speaks Bedrock's Converse API through the AWS SDK.
`endpoint::BedrockEndpoint` instead sends Rig's Anthropic Messages wire (for
Claude) and its OpenAI Chat Completions and Responses wires to Bedrock's
`bedrock-runtime` or `bedrock-mantle` endpoint. Requests are SigV4-signed with
AWS credentials, or carry a Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`).

```rust
use rig_bedrock::endpoint::{BedrockEndpoint, Host, MANTLE_CLAUDE_SONNET_5, MANTLE_GPT_OSS_120B};

let bedrock = BedrockEndpoint::from_env(Host::Mantle, rig_reqwest::shared()).await?;
let claude = bedrock.messages(MANTLE_CLAUDE_SONNET_5);
let gpt = bedrock.openai().responses(MANTLE_GPT_OSS_120B);
```

On `bedrock-mantle` most OpenAI-compatible models are served under `/v1`, the
default, and some (OpenAI's GPT models among them) under `/openai/v1`: use
`bedrock.openai_at("/openai/v1")` for those. Bedrock keeps Responses for 30
days unless a request sets `store: false`, and `bedrock-mantle` rejects
Anthropic structured output (`output_config.format`). See
[`examples/agent_with_bedrock_mantle.rs`](examples/agent_with_bedrock_mantle.rs).
