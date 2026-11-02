//! Copilot's dialect of the OpenAI wires. Completion selects the
//! Responses route for Codex model identifiers and Chat Completions
//! otherwise; every request carries the editor envelope. Credentials must be
//! exchanged before encoding (see [`auth`](super::auth)).
//!
//! ```no_run
//! use rig_core::providers::copilot::{self, DIALECT};
//! use rig_core::providers::openai::OpenAIConfig;
//!
//! # fn run(http: rig_core::http_client::DynHttpClient) {
//! let github = OpenAIConfig::with_key(&DIALECT, "tid=session-token").connect(http);
//! let model = github.completion(copilot::GPT_4O);
//! # let _ = model;
//! # }
//! ```

use crate::completion::CompletionRequest;
use crate::model::ModelInfo;
use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::providers::openai::wire::{
    Dialect, DialectHooks, EmbeddingQuirks, OpenAIConfig, Quirks, ResponsesQuirks, Route,
};
use crate::wire::Body;

use super::PROVIDER_NAME;

/// The reply header carrying Copilot's transport request id. Copilot relays
/// OpenAI's wire on both routes, header included.
pub(super) const REQUEST_ID_HEADER: Option<&str> = Some("x-request-id");

/// Copilot dialect with model-based routing and editor-identity headers.
/// Responses requests place system messages in `input`. Verification uses
/// token exchange rather than a dedicated endpoint.
pub const DIALECT: Dialect = Dialect {
    base_url_env: Some("GITHUB_COPILOT_API_BASE"),
    request_id_header: REQUEST_ID_HEADER,
    quirks: Quirks {
        hooks: Some(&HOOKS),
        verify_path: "",
        // Copilot bills a subscription, not the catalog's API rates.
        catalog_priced: false,
        base_url_env_alias: Some("COPILOT_BASE_URL"),
        // Copilot keeps no files, so no file id resolves there.
        accepts_file_ids: false,
        short_lived_key: true,
        embedding: EmbeddingQuirks {
            requires_usage: false,
            ..EmbeddingQuirks::openai()
        },
        responses: ResponsesQuirks {
            strict_tools_by_default: true,
            system_instructions: SystemInstructionsPlacement::InputSystemMessages,
            ..ResponsesQuirks::openai()
        },
        ..Quirks::openai()
    },
    ..Dialect::gateway(
        PROVIDER_NAME,
        super::GITHUB_COPILOT_API_BASE_URL,
        "GITHUB_COPILOT_API_KEY",
    )
};

static HOOKS: DialectHooks = DialectHooks {
    default_endpoint: Some(super::auth::base_url_from_token),
    model_route: Some(|model| {
        if routes_through_responses(model) {
            Route::Responses
        } else {
            Route::Chat
        }
    }),
    completion_envelope: Some(completion_envelope),
    // Non-conversational modality requests use a user initiator.
    modality_envelope: Some(|provider, request| {
        stamp(request, provider.api_key.expose(), "user", false)
    }),
};

/// The editor envelope of a completion request: the initiator and vision
/// flag follow the request's history.
fn completion_envelope(
    provider: &OpenAIConfig,
    request: &CompletionRequest,
    mut builder: http::request::Builder,
) -> http::request::Builder {
    for (name, value) in super::default_headers(
        provider.api_key.expose(),
        super::request_initiator(request),
        super::request_has_vision(request),
    ) {
        if let Some(headers) = builder.headers_mut() {
            headers.remove(name);
        }
        builder = builder.header(name, value);
    }
    builder
}

/// Choose the API route Copilot advertises for `model`.
///
/// Responses is preferred when listed. Return `None` when the catalogue
/// advertises no OpenAI-compatible completion endpoint.
pub fn route_for_model_info(model: &ModelInfo) -> Option<Route> {
    match model.supported_endpoints.as_deref() {
        Some(endpoints) => route_from_supported_endpoints(endpoints),
        None => Some(if routes_through_responses(&model.id) {
            Route::Responses
        } else {
            Route::Chat
        }),
    }
}

/// Resolve Copilot's advertised endpoint paths to a Rig completion route.
pub fn route_from_supported_endpoints(endpoints: &[String]) -> Option<Route> {
    if endpoints
        .iter()
        .any(|endpoint| endpoint.starts_with('/') && endpoint.ends_with("/responses"))
    {
        Some(Route::Responses)
    } else if endpoints
        .iter()
        .any(|endpoint| endpoint.starts_with('/') && endpoint.ends_with("/chat/completions"))
    {
        Some(Route::Chat)
    } else {
        None
    }
}

/// Return whether `model` implies the Responses API when endpoint metadata is absent.
pub fn routes_through_responses(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("codex") || model.starts_with("gpt-6.1-")
}

/// Stamp Copilot's request envelope onto a built request.
///
/// Used by the modality hook and the catalogue wire. `insert` replaces the
/// shared authentication header rather than appending a second credential.
/// Completion routes use `completion_envelope` during encoding instead.
pub(super) fn stamp(
    request: &mut http::Request<Body>,
    api_key: &str,
    initiator: &'static str,
    has_vision: bool,
) -> Result<(), http::Error> {
    let map = request.headers_mut();
    for (name, value) in super::default_headers(api_key, initiator, has_vision) {
        map.insert(
            http::HeaderName::from_bytes(name.as_bytes())?,
            http::HeaderValue::from_str(&value)?,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests;
