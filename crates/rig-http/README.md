# rig-http

Transport contracts shared by Rig's providers, HTTP middleware and transport
implementations. This crate contains no HTTP client or websocket backend and
builds for native and browser WASM targets.

- `http_client`: HTTP requests, lazy response bodies, errors and middleware.
- `http_client::framing`: incremental SSE and NDJSON parsers.
- `http_client::multipart`: transport-independent multipart forms.
- `wasm_compat`: portable bounds, boxed futures and timers.
- `ws_client` (feature `websocket`): websocket connections and frames.

`rig-reqwest` supplies the bundled HTTP implementation; `rig-tungstenite`
supplies native websocket connections. Provider configuration and model calls
belong to `rig-core`, not to the transport implementations.
