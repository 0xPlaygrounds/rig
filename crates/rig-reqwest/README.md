# rig-reqwest

The bundled [`reqwest`](https://docs.rs/reqwest) HTTP transport for [Rig](https://crates.io/crates/rig): the `HttpClientExt` implementation for `reqwest::Client` (and `reqwest_middleware::ClientWithMiddleware`) and the construction conveniences (`DefaultTransportClient`, `DefaultTransportBuilder`) that build rig-core's erased default transport over it, so `rig::providers::openai::Client::from_env()` works with no transport named.

`rig-core` itself has no default transport and no reqwest/tokio dependency; this crate is where both live. It also works when the caller has no tokio runtime (Bevy task pools, smol, `futures::executor`): reqwest futures are driven on a lazily started fallback runtime.
