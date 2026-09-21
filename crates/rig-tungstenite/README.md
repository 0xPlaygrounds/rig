# rig-tungstenite

The bundled [`tokio-tungstenite`](https://docs.rs/tokio-tungstenite) websocket backend for [Rig](https://crates.io/crates/rig): a `WebSocketClientExt` implementation, plus the default-backend conveniences (`DefaultWebSocketClient`, `DefaultWebSocketBuilder`) that let `client.responses_websocket("gpt-5.4")` resolve with no backend named.

`rig-core` owns the websocket *protocol* — the OpenAI Responses session, its event envelopes and its turn state machine all live in `rig_core::providers::openai::responses_api::websocket`, written against the transport-agnostic `rig_core::ws_client` contract. This crate owns only the socket, exactly as `rig-reqwest` owns only the HTTP transport. A second provider's websocket support is a module in rig-core, not another crate.

It also works when the caller has no tokio runtime (Bevy task pools, smol, `futures::executor`): the socket moves onto a lazily started fallback runtime and the connection becomes a pair of `futures` channels. The connection actor selects between inbound frames and commands so an idle receive cannot block a later close after cancellation. It restores undelivered frames and errors to the front of its queue when a receive is cancelled. Polling the socket also drives automatic pong replies; read-ahead is bounded to apply backpressure for slow consumers.
