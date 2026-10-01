//! OpenCode Zen and Go clients for Chat Completions, Responses, and Messages.
//!
//! [`new`] and [`from_env`] use Zen's `OPENCODE_API_KEY` and optional
//! `OPENCODE_BASE_URL`; [`go_new`] and [`go_from_env`] use Go's
//! `OPENCODE_GO_API_KEY` and optional `OPENCODE_GO_BASE_URL`. Select `.chat()`
//! or `.responses()` according to the model's endpoint in the [Zen] or [Go]
//! catalog. [`anthropic_new`] and [`go_anthropic_new`] build Messages clients.
//! Model listing returns the public catalog; credential verification is unsupported.
//!
//! Go callers should identify their application and send a stable
//! `x-opencode-session` header for each conversation. Configure those headers
//! on the transport, keeping the same session ID throughout that conversation.
//!
//! ```no_run
//! use rig_core::providers::opencode;
//! use rig_reqwest::{ReqwestClient, reqwest};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut headers = reqwest::header::HeaderMap::new();
//! headers.insert("x-opencode-session", "conversation-123".parse()?);
//! let http = ReqwestClient::from(reqwest::Client::builder()
//!     .user_agent("my-coding-agent/1.0")
//!     .default_headers(headers)
//!     .build()?);
//! let client = opencode::go_from_env()?.with_http(http);
//! let model = client.chat("kimi-k2.6");
//! # let _ = model;
//! # Ok(())
//! # }
//! ```
//!
//! [Zen]: https://opencode.ai/docs/zen/#endpoints
//! [Go]: https://opencode.ai/docs/go/#endpoints

/// OpenCode Zen's API root, including the protocol version.
pub const ZEN_API_BASE_URL: &str = "https://opencode.ai/zen/v1";

/// OpenCode Go's API root, including the protocol version.
pub const GO_API_BASE_URL: &str = "https://opencode.ai/zen/go/v1";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::OPENCODE_ZEN, "OpenCode Zen");
crate::client::macros::openai_vendor!(
    crate::providers::openai::wire::OPENCODE_GO,
    "OpenCode Go",
    go_from_env,
    go_new
);
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::OPENCODE_ZEN,
    "OpenCode Zen",
    anthropic_from_env,
    anthropic_new
);
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::OPENCODE_GO,
    "OpenCode Go",
    go_anthropic_from_env,
    go_anthropic_new
);

#[cfg(test)]
mod tests;
