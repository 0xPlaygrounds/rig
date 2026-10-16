# rig-tungstenite

The bundled [`tokio-tungstenite`](https://docs.rs/tokio-tungstenite) websocket backend for [Rig](https://crates.io/crates/rig): `TungsteniteClient`, a `WebSocketClientExt` implementation. It depends only on `rig-http`, which holds the websocket contract.

`rig-core` owns the websocket *protocol*: the OpenAI Responses session, its event envelopes and its turn state machine all live in `rig_core::providers::openai::responses_api::websocket`, written against the transport-agnostic `rig_http::ws_client` contract (re-exported at `rig_core::ws_client`). This crate owns only the socket, exactly as `rig-reqwest` owns only the HTTP transport. With rig-core's `tungstenite` feature, a session opens over this backend with no backend named:

```rust,ignore
let session = model.responses_websocket().connect().await?;
```

Without it, pass any backend: `model.responses_websocket().connect_with(&TungsteniteClient::new())`.

It also works when the caller has no tokio runtime (Bevy task pools, smol, `futures::executor`): the caller still owns and polls the socket, and each operation enters the context of `rig_http::runtime`, the lazily started fallback runtime `rig-reqwest` uses too. A cancelled receive leaves unread frames in the socket, and dropping the connection closes it. Reading on demand is the backpressure for slow consumers.
