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

/// Return whether `model` contains `codex`, case-insensitively, selecting `/responses`.
pub fn routes_through_responses(model: &str) -> bool {
    model.to_ascii_lowercase().contains("codex")
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
