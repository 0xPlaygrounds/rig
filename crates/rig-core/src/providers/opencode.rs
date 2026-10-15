//! OpenCode Zen and OpenCode Go, OpenCode's model gateways.
//!
//! [`from_env`] and [`new`] build a Zen client on the
//! [`OPENCODE`](crate::providers::openai::wire::OPENCODE) dialect, and
//! [`go_from_env`] and [`go_new`] a Go client on
//! [`OPENCODE_GO`](crate::providers::openai::wire::OPENCODE_GO). Their
//! completion sends GPT, Grok and Muse Spark models to `/responses` and every
//! other model to `/chat/completions`. The models a gateway serves over
//! Anthropic's Messages format, Claude among them, take the Messages client
//! from [`anthropic_from_env`] or [`go_anthropic_from_env`]. Every
//! constructor reads `OPENCODE_API_KEY`. Go routes by the conversation a
//! request names in `x-opencode-session`: the request's
//! [`cache_key`](crate::completion::GenerationOptions::cache_key), else a
//! fresh id. Go also asks each application to name itself in `user-agent`,
//! which a client built on its own transport (`with_http`) can send.
//!
//! ```no_run
//! use rig_core::providers::opencode;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let kimi = opencode::go_from_env()?.completion("kimi-k3");
//! let claude = opencode::anthropic_from_env()?.completion("claude-sonnet-5-5");
//! # let _ = (kimi, claude);
//! # Ok(())
//! # }
//! ```

use crate::completion::CompletionRequest;
use crate::providers::openai::wire::{DialectHooks, OpenAIConfig, Route};

/// Zen's provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "opencode";

/// Go's provider key.
pub const GO_PROVIDER_NAME: &str = "opencode-go";

/// The environment variable both gateways read their key from.
pub const API_KEY_ENV: &str = "OPENCODE_API_KEY";

/// Zen's OpenAI-format base URL.
pub const API_BASE_URL: &str = "https://opencode.ai/inference/v1";

/// Go's OpenAI-format base URL.
pub const GO_API_BASE_URL: &str = "https://opencode.ai/inference/go/v1";

/// The header Go reads a request's conversation from.
pub const SESSION_HEADER: &str = "x-opencode-session";

/// Whether the gateways serve `model` on `/responses`, as their endpoint
/// tables list the GPT, Grok and Muse Spark models.
pub fn routes_through_responses(model: &str) -> bool {
    ["gpt-", "grok-", "muse-spark-"]
        .iter()
        .any(|family| model.starts_with(family))
}

fn model_route(model: &str) -> Route {
    if routes_through_responses(model) {
        Route::Responses
    } else {
        Route::Chat
    }
}

/// Name the request's conversation, which Go requires on every completion.
fn session(
    _: &OpenAIConfig,
    request: &CompletionRequest,
    builder: http::request::Builder,
) -> http::request::Builder {
    let session = request
        .options
        .cache_key
        .clone()
        .unwrap_or_else(crate::id::generate);
    builder.header(SESSION_HEADER, session)
}

/// Zen's model routing.
pub(crate) static HOOKS: DialectHooks = DialectHooks {
    default_endpoint: None,
    model_route: Some(model_route),
    completion_envelope: None,
    modality_envelope: None,
};

/// Go's model routing and session header.
pub(crate) static GO_HOOKS: DialectHooks = DialectHooks {
    default_endpoint: None,
    model_route: Some(model_route),
    completion_envelope: Some(session),
    modality_envelope: None,
};

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
