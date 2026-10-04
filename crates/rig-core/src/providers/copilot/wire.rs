//! Copilot configuration and completion, embedding, and catalogue wires.
//! Completion selects the Responses route for Codex model identifiers and
//! Chat Completions otherwise. Credentials must be exchanged before encoding.
//!
//! ```no_run
//! use rig_core::providers::copilot::{Copilot, GPT_4O};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut model = Copilot::from_env()?.completion(GPT_4O);
//! model.wire = model.wire.with_edits_intent();
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::operation::Completion;
use crate::providers::openai::responses_api::SystemInstructionsPlacement;
/// Copilot's embeddings wire is the shared one, pointed at Copilot by
/// [`Copilot::embedding`](crate::providers::copilot::Copilot::embedding); the editor envelope is the dialect's modality
/// hook.
pub use crate::providers::openai::wire::Embeddings;
use crate::providers::openai::wire::{
    Dialect, DialectHooks, EmbeddingQuirks, OpenAIConfig, OpenAiDecoder, OpenAiWire, Quirks,
    ResponsesQuirks, Route,
};
use crate::wire::{Body, Descriptor, Encoded, Mode, Secret, Wire};

use super::{CopilotIntent, PROVIDER_NAME};

/// The reply header carrying Copilot's transport request id. Copilot relays
/// OpenAI's wire on both routes, header included.
pub(super) const REQUEST_ID_HEADER: Option<&str> = Some("x-request-id");

/// Credential variable named in missing-credential errors.
const PRIMARY_API_KEY_ENV: &str = "GITHUB_COPILOT_API_KEY";

/// Session-token variables in precedence order.
const API_KEY_ENV: [&str; 2] = ["GITHUB_COPILOT_API_KEY", "COPILOT_API_KEY"];

/// The base-URL override, in order of precedence.
const BASE_URL_ENV: &[&str] = &["GITHUB_COPILOT_API_BASE", "COPILOT_BASE_URL"];

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
    completion_envelope: Some(|provider, request, builder| {
        completion_envelope(provider, request, builder, CopilotIntent::default())
    }),
    // Non-conversational modality requests use panel intent and a user initiator.
    modality_envelope: Some(|provider, request| {
        stamp(
            request,
            provider.api_key.expose(),
            "user",
            false,
            CopilotIntent::Panel,
        )
    }),
};

/// The same envelope calculation serves the dialect hook and the public wrapper.
fn completion_envelope(
    provider: &OpenAIConfig,
    request: &CompletionRequest,
    mut builder: http::request::Builder,
    intent: CopilotIntent,
) -> http::request::Builder {
    for (name, value) in super::default_headers(
        provider.api_key.expose(),
        super::request_initiator(request),
        super::request_has_vision(request),
        intent,
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

/// Copilot's configuration: plain data, credential redacted.
///
/// The credential is an *exchanged* Copilot session token, not a GitHub
/// OAuth token: see the module docs and [`Self::from_auth`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopilotConfig {
    /// The exchanged session token. Never serialized (see [`Secret`]).
    pub api_key: Secret,
    /// The API root every path resolves against.
    pub base_url: String,
}

impl CopilotConfig {
    /// Configure Copilot with an exchanged session token.
    /// Derive a permitted endpoint from `proxy-ep=` when present, otherwise use
    /// the default. Explicit base-URL settings override token-derived routing.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        let provider = OpenAIConfig::with_key(&DIALECT, api_key);
        Self {
            api_key: provider.api_key,
            base_url: provider.base_url,
        }
    }

    /// Configure Copilot from an exchanged auth context.
    /// The context's API base takes precedence over token-derived routing.
    pub fn from_auth(context: &super::auth::AuthContext) -> Self {
        let mut provider = Self::new(context.api_key.clone());
        if let Some(api_base) = &context.api_base {
            provider.base_url = api_base.clone();
        }
        provider
    }

    /// Read an exchanged token from `GITHUB_COPILOT_API_KEY` or `COPILOT_API_KEY`.
    /// Read the optional base URL from `GITHUB_COPILOT_API_BASE` or `COPILOT_BASE_URL`.
    /// Earlier nonblank variables take precedence. Return an error for missing
    /// credentials or invalid environment values; this does not perform token exchange.
    pub fn from_env() -> Result<Self, EnvError> {
        let Some(api_key) = first_env(&API_KEY_ENV)? else {
            return Err(EnvError::Variable {
                name: PRIMARY_API_KEY_ENV,
                source: std::env::VarError::NotPresent,
            });
        };
        let mut provider = Self::new(api_key);
        if let Some(base_url) = first_env(BASE_URL_ENV)? {
            provider.base_url = base_url;
        }
        Ok(provider)
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// The completion wire for `model`, on whichever route answers it.
    pub(crate) fn completion(&self, model: impl Into<String>) -> CopilotWire {
        CopilotWire {
            wire: self.openai().completion(model),
            intent: CopilotIntent::default(),
        }
    }

    /// Build an embedding wire with Copilot's editor headers and optional usage.
    /// Use `ndims` when supplied, otherwise the shared wire's model default.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        Embeddings::new(self.openai(), model, ndims)
    }

    /// Convert to shared configuration with Copilot's dialect and explicit endpoint.
    fn openai(&self) -> OpenAIConfig {
        OpenAIConfig::with_key(&DIALECT, self.api_key.clone()).with_base_url(self.base_url.clone())
    }

    /// Resolve `path` against the base URL.
    pub(super) fn uri(&self, path: &str) -> String {
        format!("{}{path}", self.base_url.trim_end_matches('/'))
    }
}

/// Read the first nonblank environment variable in `names`, preserving its value.
/// Return an error if an encountered variable cannot be decoded.
fn first_env(names: &[&'static str]) -> Result<Option<String>, EnvError> {
    for name in names {
        if let Some(value) = env::optional(name)?.filter(|value| !value.trim().is_empty()) {
            return Ok(Some(value));
        }
    }
    Ok(None)
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
    intent: CopilotIntent,
) -> Result<(), http::Error> {
    let map = request.headers_mut();
    for (name, value) in super::default_headers(api_key, initiator, has_vision, intent) {
        map.insert(
            http::HeaderName::from_bytes(name.as_bytes())?,
            http::HeaderValue::from_str(&value)?,
        );
    }
    Ok(())
}

/// Completion wire with Copilot's conversation intent and editor headers.
/// Delegates payload handling to `wire`, but overrides its request envelope
/// even when that field contains another dialect.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopilotWire {
    /// The route's wire, pointed at Copilot.
    pub wire: OpenAiWire,
    /// The conversation intent this turn declares (`openai-intent`).
    pub intent: CopilotIntent,
}

impl CopilotWire {
    /// The conversation intent this wire declares.
    pub fn intent(&self) -> CopilotIntent {
        self.intent
    }

    /// Declare `intent` in the `openai-intent` header.
    pub fn with_intent(mut self, intent: CopilotIntent) -> Self {
        self.intent = intent;
        self
    }

    /// Declare the generic chat-panel conversation semantics.
    pub fn with_panel_intent(self) -> Self {
        self.with_intent(CopilotIntent::Panel)
    }

    /// Declare the edit-oriented conversation semantics.
    pub fn with_edits_intent(self) -> Self {
        self.with_intent(CopilotIntent::Edits)
    }

    /// Sanitize tool schemas for strict mode on whichever route answers.
    ///
    /// The shared [`Responses::new`](crate::providers::openai::responses_api::wire::Responses::new)
    /// constructor already enables this for Copilot, so this is the chat route's opt-in.
    pub fn with_strict_tools(mut self) -> Self {
        self.wire = self.wire.with_strict_tools();
        self
    }

    /// Serialize tool-result content as arrays.
    ///
    /// A chat-completions shape: the Responses request has one content
    /// encoding, so this is a no-op on that route.
    pub fn with_tool_result_array_content(mut self) -> Self {
        if let OpenAiWire::Chat(wire) = self.wire {
            self.wire = OpenAiWire::Chat(wire.with_tool_result_array_content());
        }
        self
    }
}

impl Wire for CopilotWire {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = OpenAiDecoder;

    fn describe(&self) -> Descriptor<'_> {
        self.wire.describe()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        self.wire
            .encode_with_headers(request, mode, |provider, request, builder| {
                completion_envelope(provider, request, provider.headers(builder), self.intent)
            })
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        self.wire.decoder()
    }
}

#[cfg(test)]
mod tests;
