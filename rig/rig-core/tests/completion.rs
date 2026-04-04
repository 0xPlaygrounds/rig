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
//!   - `OPENROUTER_API_KEY`
//!   - `MOONSHOT_API_KEY`
//!   - `COHERE_API_KEY`
//!   - `DEEPSEEK_API_KEY`
//!   - `TOGETHER_API_KEY`
//!   - `XAI_API_KEY`
//!   - `HUGGINGFACE_API_KEY`
//!   - `GALADRIEL_API_KEY`
//!   - `MIRA_API_KEY`
//!
//! Run a single provider smoke test with:
//!   - `cargo test -p rig-core --test completion <module>::completion_smoke -- --ignored --nocapture`
//!
//! Run the full completion smoke target:
//!   - export all required API keys in the shell first
//!   - `cargo test -p rig-core --test completion -- --ignored --nocapture`
//!
//! Missing environment variables fail fast through each provider client's
//! `from_env()` path instead of skipping the targeted test.

#[path = "common/support.rs"]
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

#[path = "completion/openrouter_agent.rs"]
mod openrouter_agent;

#[path = "completion/moonshot_agent.rs"]
mod moonshot_agent;

#[path = "completion/cohere_agent.rs"]
mod cohere_agent;

#[path = "completion/deepseek_agent.rs"]
mod deepseek_agent;

#[path = "completion/together_agent.rs"]
mod together_agent;

#[path = "completion/xai_agent.rs"]
mod xai_agent;

#[path = "completion/huggingface_agent.rs"]
mod huggingface_agent;

#[path = "completion/galadriel_agent.rs"]
mod galadriel_agent;

#[path = "completion/mira_agent.rs"]
mod mira_agent;
