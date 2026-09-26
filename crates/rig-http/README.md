# rig-http

The transport contracts every [Rig](https://github.com/0xPlaygrounds/rig)
provider sends through:

- `http_client`: `HttpClientExt`, the trait an HTTP transport implements,
  `DynHttpClient` (an erased client with middleware), request framing,
  multipart forms and the transport error type;
- `ws_client` (feature `websocket`): the websocket connection contract the
  Responses WebSocket session is written against;
- `wasm_compat`: the `Send`/`Sync` bounds and boxed futures that relax on
  browser wasm.

This crate has no transport of its own. `rig-reqwest` (HTTP) and
`rig-tungstenite` (websocket) are the bundled transports, and depend only on
this crate. `rig-core` re-exports these modules at `rig_core::http_client`,
`rig_core::ws_client` and `rig_core::wasm_compat`, so provider code and
users reach them there.

A transport author depends on `rig-http` alone and implements
`HttpClientExt`:

```rust
use rig_http::http_client::{HttpClientExt, DynHttpClient};

fn erase(client: impl HttpClientExt + Clone + 'static) -> DynHttpClient {
    DynHttpClient::new(client)
}
```

The `test-utils` feature adds HTTP client doubles for tests.
