# rig-tungstenite

The bundled [`tokio-tungstenite`](https://docs.rs/tokio-tungstenite) websocket backend for [Rig](https://crates.io/crates/rig): `TungsteniteClient`, a `WebSocketClientExt` implementation. It depends only on `rig-http`, which holds the websocket contract.

`rig-core` owns the websocket *protocol*: the OpenAI Responses websocket wire and transport, and the turn lifecycle, live in `rig_core::providers::openai::responses_api::websocket`, written against the transport-agnostic `rig_http::ws_client` contract (re-exported at `rig_core::ws_client`). This crate owns only the socket, exactly as `rig-reqwest` owns only the HTTP transport. With rig-core's `tungstenite` feature, a websocket model opens over this backend with no backend named:

```rust,ignore
let socket = model.responses_websocket().connect().await?;
let response = socket.call("Hello").await?;
```

Without it, pass any backend: `model.responses_websocket().connect_with(&TungsteniteClient::new())`.

It also works when the caller has no tokio runtime (Bevy task pools, smol, `futures::executor`): the socket moves onto a lazily started fallback runtime and the connection becomes a pair of `futures` channels. The connection actor selects between inbound frames and commands so an idle receive cannot block a later close after cancellation. It restores undelivered frames and errors to the front of its queue when a receive is cancelled. Polling the socket also drives automatic pong replies; read-ahead is bounded to apply backpressure for slow consumers.
