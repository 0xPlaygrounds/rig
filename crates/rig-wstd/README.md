# rig-wstd

The WASI HTTP transport for [Rig](https://crates.io/crates/rig): `WstdClient` implements `HttpClientExt` over the `wasi:http` client of [`wstd`](https://docs.rs/wstd), so a provider client runs inside a WASI 0.2 component (`wasm32-wasip2`). It depends only on `rig-http`, which holds that contract. On targets other than WASI the crate is empty.

A WASI build leaves out the bundled reqwest transport, which needs sockets and threads WASI does not have, and names this one:

```rust
use rig_core::providers::openai::{self, OpenAIConfig};
use rig_wstd::WstdClient;

#[wstd::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let openai = OpenAIConfig::from_env()?.connect(WstdClient::default());
    let model = openai.completion(openai::GPT_4O);
    Ok(())
}
```

wstd's reactor drives every request, response body and stream, so await them inside the wstd runtime (`#[wstd::main]` or `wstd::runtime::block_on`). `WstdClient::default()` uses the host's timeouts; `WstdClient::from(client)` wraps a `wstd::http::Client` with its own. The host must grant outgoing HTTP, for example `wasmtime run -S http --env OPENAI_API_KEY app.wasm`. The `wasi_completion` example streams an agent prompt this way.
