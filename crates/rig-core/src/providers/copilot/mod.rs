//! GitHub Copilot: the credential exchange, the request envelope, and the
//! model identifiers.
//!
//! Copilot serves Chat Completions, Responses and Embeddings behind
//! `https://api.githubcopilot.com`, and [`wire`] is all three:
//! [`wire::Copilot`] is the configuration, [`wire::CopilotWire`] is the
//! completion wire that picks the route ([`wire::routes_through_responses`]
//! is the only home of that predicate — Codex-class models are answered by
//! `/responses`, everything else by `/chat/completions`), and
//! [`wire::Embeddings`] is the shared embeddings wire under Copilot's
//! envelope, while [`wire::Models`] is the one endpoint whose reply really is
//! Copilot's own — it names the vendor behind each model and nests the
//! modality under `capabilities.type`.
//!
//! Two things Copilot cannot express as a wire stay here. The credential is
//! *exchanged* over the network before the API can be called at all — a
//! device-code poll, a refresh, a shared token cache — and a synchronous
//! encode has no seat for a round trip, so the exchange is [`auth`] and
//! [`wire::Copilot::from_auth`] is the bridge. And the editor identity every
//! request carries (`copilot-integration-id`, `editor-version`,
//! `openai-intent`, `X-Initiator`, …) is stamped onto the request whichever
//! wire built it, so the header set has one definition — a crate-internal
//! `default_headers` beside the model identifiers — rather than one per route.
//!
//! # Example
//! ```no_run
//! use rig_core::providers::copilot;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // An already-exchanged session token, from the environment; the wires,
//! // which `.bind(transport)` joins to a socket.
//! let github = copilot::wire::Copilot::from_env()?;
//!
//! // The model picks the route; the caller does not.
//! let chat = github.completion(copilot::GPT_4O);
//! let codex = github.completion(copilot::GPT_5_3_CODEX);
//! let embeddings = github.embeddings(copilot::TEXT_EMBEDDING_3_SMALL, None);
//! let catalogue = github.models();
//! # let _ = (chat, codex, embeddings, catalogue);
//! # Ok(())
//! # }
//! ```
//!
//! Signing in, rather than reading an exchanged token, runs [`auth`] first:
//! ```no_run
//! use rig_core::providers::copilot::{auth, wire};
//!
//! # async fn run<H: rig_core::http_client::HttpClientExt>(http: H) -> Result<(), Box<dyn std::error::Error>> {
//! let authenticator = auth::Authenticator::new(
//!     auth::AuthSource::OAuth,
//!     auth::default_token_dir().map(|dir| dir.join("access-token")),
//!     auth::default_token_dir().map(|dir| dir.join("api-key.json")),
//!     auth::DeviceCodeHandler::default(),
//!     true,
//! );
//! let context = authenticator.auth_context(&http).await?;
//! let github = wire::Copilot::from_auth(&context);
//! # let _ = github;
//! # Ok(())
//! # }
//! ```

/// The credential exchange. Public because the wire configuration holds an
/// *exchanged* session token ([`wire::Copilot::from_auth`]) and the exchange
/// — a device-code poll, a refresh, a shared token cache — is a
/// conversation, so it cannot happen inside a wire's synchronous `encode`.
pub mod auth;
pub mod wire;

use crate::completion;
use crate::providers::openai;
use serde::{Deserialize, Serialize};

/// The API root a token that names no endpoint of its own resolves against.
pub(crate) const GITHUB_COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
pub(crate) const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.35.0";
pub(crate) const USER_AGENT: &str = "GitHubCopilotChat/0.35.0";
pub(crate) const EDITOR_VERSION: &str = "vscode/1.107.0";
const API_VERSION: &str = "2025-04-01";

/// Copilot's catalogue endpoint. `GET /models` answers with the whole
/// catalogue — there is no cursor — so [`wire::Models`] needs no query.
pub(crate) const MODEL_LISTING_PATH: &str = "/models";

/// Stable descriptor name reported on normalized Copilot responses.
pub const PROVIDER_NAME: &str = "copilot";

/// Copilot conversation intent sent in the `openai-intent` request header.
///
/// Serialized because it is a field of [`wire::CopilotWire`], which is data
/// a host may store.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CopilotIntent {
    #[default]
    Panel,
    Edits,
}

impl CopilotIntent {
    pub fn as_header(self) -> &'static str {
        match self {
            CopilotIntent::Panel => "conversation-panel",
            CopilotIntent::Edits => "conversation-edits",
        }
    }
}

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

/// The encoding [`wire::Embeddings::with_encoding_format`] asks for. Copilot
/// relays OpenAI's embeddings wire, so it is OpenAI's own type.
pub use openai::EncodingFormat;

/// Copilot's editor envelope: the headers every request on every route
/// carries. Without them the API answers 400 regardless of the body.
///
/// One definition, applied by [`wire`] to the finished request, because two
/// of the three routes are shared wires Copilot does not own.
pub(crate) fn default_headers(
    api_key: &str,
    initiator: &'static str,
    has_vision: bool,
    intent: CopilotIntent,
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
        ("openai-intent", intent.as_header().to_string()),
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

/// Derive the Copilot REST base URL from a chat token's `proxy-ep=` segment.
///
/// The endpoint is parsed from a credential string, not from explicit caller
/// configuration. For that reason, token-derived routing is limited to GitHub
/// Copilot service hosts and HTTPS. Callers that need a custom non-GitHub host
/// can still opt in explicitly with [`wire::Copilot::with_base_url`].
pub(crate) fn base_url_from_token(token: &str) -> Option<String> {
    let proxy_ep = token
        .split(';')
        .find_map(|part| part.trim().strip_prefix("proxy-ep="))?
        .trim();

    normalize_copilot_proxy_endpoint(proxy_ep)
}

fn normalize_copilot_proxy_endpoint(proxy_ep: &str) -> Option<String> {
    if proxy_ep.is_empty() {
        return None;
    }

    let candidate = if proxy_ep.starts_with("http://") || proxy_ep.starts_with("https://") {
        proxy_ep.to_string()
    } else {
        format!("https://{proxy_ep}")
    };

    let mut url = url::Url::parse(&candidate).ok()?;
    if url.scheme() != "https" || !url.username().is_empty() || url.password().is_some() {
        return None;
    }
    if url.path() != "/" || url.query().is_some() || url.fragment().is_some() {
        return None;
    }

    let host = url.host_str()?.to_ascii_lowercase();
    if !is_allowed_token_derived_copilot_host(&host) {
        return None;
    }

    let api_host = host
        .strip_prefix("proxy.")
        .map(|suffix| format!("api.{suffix}"))
        .unwrap_or(host);
    url.set_host(Some(&api_host)).ok()?;

    Some(url.to_string().trim_end_matches('/').to_string())
}

fn is_allowed_token_derived_copilot_host(host: &str) -> bool {
    host == "githubcopilot.com" || host.ends_with(".githubcopilot.com")
}

/// The `X-Initiator` this request declares: `agent` once the turn contains
/// anything the model itself produced, `user` otherwise.
pub(crate) fn request_initiator(request: &completion::CompletionRequest) -> &'static str {
    for message in request.chat_history.iter() {
        match message {
            crate::completion::Message::Assistant { .. } => return "agent",
            crate::completion::Message::User { content } => {
                if content
                    .iter()
                    .any(|item| matches!(item, crate::message::UserContent::ToolResult(_)))
                {
                    return "agent";
                }
            }
            crate::completion::Message::System { .. } => {}
        }
    }

    "user"
}

/// Whether this request carries an image, which Copilot gates behind
/// `copilot-vision-request`.
pub(crate) fn request_has_vision(request: &completion::CompletionRequest) -> bool {
    request.chat_history.iter().any(|message| match message {
        crate::completion::Message::User { content } => content
            .iter()
            .any(|item| matches!(item, crate::message::UserContent::Image(_))),
        _ => false,
    })
}

/// Copilot's error body, which names the failure under either key.
#[derive(Debug, Deserialize)]
pub struct ChatApiErrorResponse {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl ChatApiErrorResponse {
    pub fn error_message(&self) -> &str {
        self.message
            .as_deref()
            .or(self.error.as_deref())
            .unwrap_or("unknown error")
    }
}

#[cfg(test)]
mod tests;
