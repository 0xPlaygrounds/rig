//! OpenCode Zen and Go: gateways that serve each model on one of three APIs,
//! Chat Completions, Responses or Anthropic's Messages.
//!
//! Which API serves a model is in the model catalog, so
//! [`Catalog::connect`](crate::catalog::Catalog::connect) reaches each one on
//! its own. The constructors here build one client per format: [`from_env`]
//! and [`new`] for Zen's Chat Completions and Responses, which
//! [`OpenAI::completion`](crate::providers::openai::OpenAI::completion)
//! chooses between by the catalog, and [`anthropic_from_env`] and
//! [`anthropic_new`] for its Messages models. A Messages model asked of the
//! first client goes to Chat Completions, which does not serve it. The `go_`
//! constructors are Go's. Both read `OPENCODE_API_KEY`. Zen's free models
//! answer only OpenCode's own client.
//!
//! Go refuses a request without an `x-opencode-session` header, which
//! carries the request's
//! [`cache_key`](crate::completion::GenerationOptions::cache_key), or a fresh
//! id without one. Set one key per conversation, so Go routes its turns
//! together and reuses their prompt cache.
//!
//! ```no_run
//! use rig_core::catalog::Catalog;
//! use rig_core::providers::opencode;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let haiku = Catalog::builtin().connect("opencode/claude-haiku-5-5")?;
//! let glm = opencode::go_from_env()?.completion("glm-5.3");
//! # let _ = (haiku, glm);
//! # Ok(())
//! # }
//! ```

/// Zen's provider key: the dialect name, a reply's `Origin::provider` and
/// the catalog vendor.
pub(crate) const ZEN: &str = "opencode";

/// Go's provider key.
pub(crate) const GO: &str = "opencode-go";

/// The header Go routes a conversation's requests together by.
pub(crate) const SESSION_HEADER: &str = "x-opencode-session";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::OPENCODE, "OpenCode Zen");
crate::client::macros::openai_vendor!(
    crate::providers::openai::wire::OPENCODE_GO,
    "OpenCode Go",
    go_from_env,
    go_new
);
crate::client::macros::anthropic_vendor!(
    crate::providers::anthropic::wire::OPENCODE,
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
