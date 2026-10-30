//! GitHub Copilot authentication, the Copilot dialect of the OpenAI wires,
//! and model identifiers.
//!
//! Copilot is an OpenAI-shaped provider: configure it with
//! `OpenAIConfig::with_key(&copilot::DIALECT, session_token)`. To log in with
//! a device code or exchange a GitHub token for a session, use an
//! [`auth::Authenticator`]; [`SignedInModel`](crate::client::signed_in::SignedInModel)
//! serves a catalog model with the session it resolves.
//! Completion routes can follow the model catalogue's advertised endpoints through
//! [`wire::route_for_model_info`]; all routes include editor-identity headers.
//!
//! ```no_run
//! use rig_core::providers::copilot;
//! use rig_core::providers::openai::OpenAIConfig;
//!
//! # fn run(http: rig_core::http_client::DynHttpClient) {
//! let github = OpenAIConfig::with_key(&copilot::DIALECT, "tid=session-token").connect(http);
//!
//! let chat = github.completion(copilot::GPT_4O);
//! let codex = github.completion(copilot::GPT_5_3_CODEX);
//! let embeddings = github.embedding(copilot::TEXT_EMBEDDING_3_SMALL, None);
//! # let _ = (chat, codex, embeddings);
//! # }
//! ```

/// Device login, session-token exchange, refresh, and the token store that keeps them.
///
/// ```no_run
/// use rig_core::providers::copilot::auth::{AuthSource, Authenticator, DeviceCodeHandler, MemoryTokenStore};
///
/// let auth = Authenticator::new(AuthSource::OAuth, MemoryTokenStore::default(), DeviceCodeHandler::default(), true);
/// ```
pub mod auth;
pub mod extension;
pub mod model_listing;
pub mod wire;

pub use wire::DIALECT;

use crate::completion;

/// The API root a token that names no endpoint of its own resolves against.
pub(crate) const GITHUB_COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
pub(crate) const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub(crate) const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub(crate) const EDITOR_VERSION: &str = "vscode/1.107.0";
const API_VERSION: &str = "2025-04-01";

/// Stable descriptor name reported on normalized Copilot responses.
pub const PROVIDER_NAME: &str = "copilot";

/// `gpt-4`
pub const GPT_4: &str = "gpt-4";
/// `gpt-4o`
pub const GPT_4O: &str = "gpt-4o";
/// `gpt-4o-mini`
pub const GPT_4O_MINI: &str = "gpt-4o-mini";
/// `gpt-4.1`
pub const GPT_4_1: &str = "gpt-4.1";
/// `gpt-4.1-mini`
pub const GPT_4_1_MINI: &str = "gpt-4.1-mini";
/// `gpt-4.1-nano`
pub const GPT_4_1_NANO: &str = "gpt-4.1-nano";
/// `gpt-5.3-codex`
pub const GPT_5_3_CODEX: &str = "gpt-5.3-codex";
/// `gpt-5.1-codex`
pub const GPT_5_1_CODEX: &str = "gpt-5.1-codex";
/// `gpt-5.5`
pub const GPT_5_5: &str = "gpt-5.5";
/// `gpt-5.4`
pub const GPT_5_4: &str = "gpt-5.4";
/// `gpt-6.1-terra`
pub const GPT_6_1_TERRA: &str = "gpt-6.1-terra";
/// `claude-sonnet-4` completion model (Anthropic, via Copilot)
pub const CLAUDE_SONNET_4: &str = "claude-sonnet-4";
/// `claude-sonnet-4.6`
pub const CLAUDE_SONNET_4_6: &str = "claude-sonnet-4.6";
/// `claude-opus-4.6`
pub const CLAUDE_OPUS_4_6: &str = "claude-opus-4.6";
/// `claude-opus-4.7`
pub const CLAUDE_OPUS_4_7: &str = "claude-opus-4.7";
/// `claude-3.5-sonnet` completion model (Anthropic, via Copilot)
pub const CLAUDE_3_5_SONNET: &str = "claude-3.5-sonnet";
/// `gemini-3-flash-preview` completion model (Google, via Copilot)
pub const GEMINI_3_FLASH: &str = "gemini-3-flash-preview";
/// `gemini-3.1-pro-preview` completion model (Google, via Copilot)
pub const GEMINI_3_1_PRO_FLASH: &str = "gemini-3.1-pro-preview";
/// `gemini-2.0-flash-001` completion model (Google, via Copilot)
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash-001";
/// `o3-mini` reasoning model (OpenAI, via Copilot)
pub const O3_MINI: &str = "o3-mini";
/// `text-embedding-3-small`
pub const TEXT_EMBEDDING_3_SMALL: &str = "text-embedding-3-small";
/// `text-embedding-3-large`
pub const TEXT_EMBEDDING_3_LARGE: &str = "text-embedding-3-large";
/// `text-embedding-ada-002`
pub const TEXT_EMBEDDING_ADA_002: &str = "text-embedding-ada-002";

/// Build required editor-identity headers, including credentials and request identity.
pub(crate) fn default_headers(
    api_key: &str,
    initiator: &'static str,
    has_vision: bool,
) -> Vec<(&'static str, String)> {
    let mut headers = vec![
        (
            http::header::AUTHORIZATION.as_str(),
            format!("Bearer {api_key}"),
        ),
        ("copilot-integration-id", "vscode-chat".to_string()),
        ("editor-version", EDITOR_VERSION.to_string()),
        ("editor-plugin-version", EDITOR_PLUGIN_VERSION.to_string()),
        ("user-agent", USER_AGENT.to_string()),
        ("openai-intent", "conversation-panel".to_string()),
        ("x-github-api-version", API_VERSION.to_string()),
        ("x-request-id", crate::id::generate()),
        (
            "x-vscode-user-agent-library-version",
            "electron-fetch".to_string(),
        ),
        ("X-Initiator", initiator.to_string()),
    ];

    if has_vision {
        headers.push(("copilot-vision-request", "true".to_string()));
    }

    headers
}

/// The `X-Initiator` this request declares: `agent` once the turn contains
/// anything the model itself produced, `user` otherwise.
pub(crate) fn request_initiator(request: &completion::CompletionRequest) -> &'static str {
    use crate::completion::Message;
    let agent = request.chat_history.iter().any(|message| match &**message {
        Message::Assistant(_) => true,
        Message::User { content } => content
            .iter()
            .any(|item| matches!(item, crate::message::UserContent::ToolResult(_))),
        Message::System { .. } => false,
    });
    if agent { "agent" } else { "user" }
}

/// Whether this request carries an image, which Copilot gates behind
/// `copilot-vision-request`.
pub(crate) fn request_has_vision(request: &completion::CompletionRequest) -> bool {
    request.chat_history.iter().any(|message| match &**message {
        crate::completion::Message::User { content } => content
            .iter()
            .any(|item| matches!(item, crate::message::UserContent::Image(_))),
        _ => false,
    })
}

#[cfg(test)]
mod tests;
