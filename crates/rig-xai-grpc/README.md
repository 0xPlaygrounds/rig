# rig-xai-grpc

This companion crate sends Rig completion requests to xAI over its gRPC API
(`api.x.ai:443`) instead of the REST API.

Requests are built by rig-core's xAI Chat Completions encoder and sent as the
`GetCompletionsRequest` they describe. Replies are read back by the same
decoder. History, tool calls, reasoning, typed options (`XaiOptions`) and
reply extras (`XaiExtras`) work as on the REST wire, and replies report the
provider as `xai`. A response's `raw` is the gRPC reply as proto3 JSON under
the proto's field names.

The crate is native only: tonic's transport does not build for WebAssembly.

## Usage

```toml
[dependencies]
rig-xai-grpc = "0.44"
rig-agent = "0.44"
```

Or enable the `xai-grpc` feature of the `rig` facade and use `rig::xai_grpc`.

Set your xAI API key:

```shell
export XAI_API_KEY=your_api_key_here
```

```rust,no_run
use rig_agent::prelude::*;
use rig_xai_grpc::XaiGrpc;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let transport = XaiGrpc::from_env().await?;
    let agent = AgentBuilder::new(transport.completion("grok-4.3"))
        .preamble("You are a helpful assistant.")
        .build();
    let response = agent.prompt("Hello!").await?;
    println!("{}", response.output());
    Ok(())
}
```

See [`examples/xai_grpc_agent.rs`](examples/xai_grpc_agent.rs) for tool calls
and streaming.

## What the request carries

Every field of `GetCompletionsRequest` can be set through
`additional_params`, spelled as proto3 JSON (`"search_parameters"`,
`"include": ["INCLUDE_OPTION_INLINE_CITATIONS"]`, or a server-side tool such
as `"tools": [{"web_search": {}}]`). A field the proto does not declare is an
error rather than dropped. The gRPC request has no prompt cache key, so a
`cache_key` option or `XaiOptions::prompt_cache_key` is refused. Audio and
video inputs have no content part in the proto.

## Protos

`proto/` holds xAI's official protos, unmodified, from
[xai-org/xai-proto](https://github.com/xai-org/xai-proto) at commit
`065692d0455244b0d5f0ccf9c05f8b98ff4618cd` (`proto/xai/api/v1/`). They are
licensed under the Apache License 2.0, a copy of which is in
[`proto/LICENSE`](proto/LICENSE). The client is generated from them at build
time with a vendored `protoc`, so no system protobuf compiler is needed.
