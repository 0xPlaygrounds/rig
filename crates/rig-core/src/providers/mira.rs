//! Mira's gateway, as a dialect.
//!
//! Mira is an OpenAI chat-completions dialect, so it has no client, no
//! models, and no response types of its own:
//! [`openai::wire::MIRA`](crate::providers::openai::wire::MIRA) carries the
//! base URL (the bare host — Mira's chat and listing paths carry their own
//! `/v1`), the `MIRA_API_KEY` variable, and the quirks that make it Mira:
//! the gateway rejects tool parameters, OpenAI structured-output
//! parameters, and unknown parameters such as `stream_options`, it accepts
//! only plain `{role, content}` string messages (names stripped,
//! content-part arrays flattened), and it checks a credential at
//! `/user-credits` rather than by listing models.
//!
//! Mira publishes no model identifiers of its own — the gateway forwards to
//! other providers' models, so a model handle is whatever `GET /v1/models`
//! reports. That is why this module declares no constants.
//!
//! # Example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::openai::wire::{MIRA, OpenAI};
//! use rig_reqwest::DefaultTransport;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = OpenAI::from_env_with(&MIRA)?.bound()?;
//! let models = provider.model_listing().list_all().await?;
//! # Ok(())
//! # }
//! ```
