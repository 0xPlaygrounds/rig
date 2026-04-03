//! Integration tests: ignored provider-backed smoke tests for one-shot agent completions.
//!
//! These tests make live provider API calls, so they are ignored by default.
//! Each test expects its provider API key to already be exported in the shell
//! environment before `cargo test` runs. This target does not auto-load `.env`
//! files.
//!
//! Required environment variables:
//!   - `OPENAI_API_KEY`
//!   - `ANTHROPIC_API_KEY`
//!   - `GEMINI_API_KEY`
//!   - `GROQ_API_KEY`
//!   - `HYPERBOLIC_API_KEY`
//!
//! Run a single provider smoke test:
//!   - `OPENAI_API_KEY=... cargo test -p rig-core --test completion openai_agent::completion_smoke -- --ignored --nocapture`
//!   - `ANTHROPIC_API_KEY=... cargo test -p rig-core --test completion anthropic_agent::completion_smoke -- --ignored --nocapture`
//!   - `GEMINI_API_KEY=... cargo test -p rig-core --test completion gemini_agent::completion_smoke -- --ignored --nocapture`
//!   - `GROQ_API_KEY=... cargo test -p rig-core --test completion groq_agent::completion_smoke -- --ignored --nocapture`
//!   - `HYPERBOLIC_API_KEY=... cargo test -p rig-core --test completion hyperbolic_agent::completion_smoke -- --ignored --nocapture`
//!
//! Run the full completion smoke target:
//!   - export all required API keys in the shell first
//!   - `cargo test -p rig-core --test completion -- --ignored --nocapture`
//!
//! Missing environment variables fail fast through each provider client's
//! `from_env()` path instead of skipping the targeted test.

#[path = "completion/support.rs"]
mod support;

#[path = "completion/openai_agent.rs"]
mod openai_agent;

#[path = "completion/anthropic_agent.rs"]
mod anthropic_agent;

#[path = "completion/gemini_agent.rs"]
mod gemini_agent;

#[path = "completion/groq_agent.rs"]
mod groq_agent;

#[path = "completion/hyperbolic_agent.rs"]
mod hyperbolic_agent;
