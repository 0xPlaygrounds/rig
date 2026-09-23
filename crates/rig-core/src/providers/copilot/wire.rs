//! Copilot configuration and completion, embedding, and catalogue wires.
//! Completion selects the Responses route for Codex model identifiers and
//! Chat Completions otherwise. Credentials must be exchanged before encoding.
//!
//! ```no_run
//! use rig_core::providers::copilot::{wire::Copilot, GPT_4O};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let wire = Copilot::from_env()?.completion(GPT_4O).with_edits_intent();
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::completion::{CompletionRequest, ProviderCapabilities};
use crate::driver::{HasEmbedding, HasModelListing};
use crate::error::EncodeError;
use crate::model::{Model, ModelList};
use crate::operation::{Completion, ModelListing};
use crate::providers::internal::wire::classify_untyped_line;
use crate::providers::openai::responses_api::SystemInstructionsPlacement;
/// Copilot's embeddings wire is the shared one, pointed at Copilot by
/// [`Copilot::embeddings`]; the editor envelope is the dialect's modality
/// hook.
pub use crate::providers::openai::wire::Embeddings;
use crate::providers::openai::wire::{
    Dialect, DialectHooks, EmbeddingQuirks, OpenAI, OpenAiDecoder, OpenAiWire, Quirks,
    ResponsesQuirks, Route,
};
use crate::telemetry::GenAiOperation;
use crate::wire::{
    Body, Decoder, Encoded, Framing, HasCompletion, Mode, Output, Secret, Sink, Wire, WireEvent,
    WireFrame,
};

use super::{CopilotIntent, PROVIDER_NAME};

/// The reply header carrying Copilot's transport request id. Copilot relays
/// OpenAI's wire on both routes, header included.
const REQUEST_ID_HEADER: Option<&str> = Some("x-request-id");

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
    default_endpoint: Some(super::base_url_from_token),
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
    provider: &OpenAI,
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
pub struct Copilot {
    /// The exchanged session token. Never serialized (see [`Secret`]).
    pub api_key: Secret,
    /// The API root every path resolves against.
    pub base_url: String,
}

impl Copilot {
    /// Configure Copilot with an exchanged session token.
    /// Derive a permitted endpoint from `proxy-ep=` when present, otherwise use
    /// the default. Explicit base-URL settings override token-derived routing.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        credential_of(&OpenAI::with_key(&DIALECT, api_key))
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
    pub fn completion(&self, model: impl Into<String>) -> CopilotWire {
        self.wire_for(model)
    }

    /// Shared construction for the inherent and trait completion entry points.
    fn wire_for(&self, model: impl Into<String>) -> CopilotWire {
        CopilotWire {
            wire: self.openai().completion(model),
            intent: CopilotIntent::default(),
        }
    }

    /// Build an embedding wire with Copilot's editor headers and optional usage.
    /// Use `ndims` when supplied, otherwise the shared wire's model default.
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        Embeddings::new(self.openai(), model, ndims)
    }

    /// The model-listing wire.
    pub fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }

    /// Convert to shared configuration with Copilot's dialect and explicit endpoint.
    fn openai(&self) -> OpenAI {
        OpenAI::with_key(&DIALECT, self.api_key.clone()).with_base_url(self.base_url.clone())
    }

    /// Resolve `path` against the base URL.
    fn uri(&self, path: &str) -> String {
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
fn stamp(
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

/// The Copilot credential behind the shared configuration
/// [`Copilot::new`] resolves its endpoint through.
fn credential_of(provider: &OpenAI) -> Copilot {
    Copilot {
        api_key: provider.api_key.clone(),
        base_url: provider.base_url.clone(),
    }
}

impl Wire for CopilotWire {
    type Op = Completion;
    type Decoder = OpenAiDecoder;

    fn name(&self) -> &str {
        self.wire.name()
    }

    fn model(&self) -> Option<&str> {
        self.wire.model()
    }

    fn replay_issuers(&self, model: Option<&str>) -> Vec<String> {
        self.wire.replay_issuers(model)
    }

    fn route(&self) -> Option<&str> {
        self.wire.route()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        self.wire
            .encode_with_headers(request, mode, |provider, request, builder| {
                completion_envelope(provider, request, provider.headers(builder), self.intent)
            })
    }

    fn decoder(&self, mode: Mode) -> OpenAiDecoder {
        self.wire.decoder(mode)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.wire.capabilities()
    }

    fn telemetry(&self, streaming: bool) -> GenAiOperation {
        self.wire.telemetry(streaming)
    }
}

/// Copilot's model-listing wire.
///
/// `GET /models` answers with the whole catalogue, so
/// [`Decoder::continuation`] keeps its default `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// Which Copilot, and how to reach it.
    pub provider: Copilot,
}

/// Catalogue entry with model identity, vendor, and modality under `capabilities.type`.
#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    vendor: Option<String>,
    #[serde(default)]
    capabilities: Option<ModelEntryCapabilities>,
}

#[derive(Debug, Deserialize)]
struct ModelEntryCapabilities {
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

/// The `{ "data": [...] }` envelope.
#[derive(Debug, Deserialize)]
pub struct ModelsReply {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

impl ModelsReply {
    /// The catalogue as normalized models.
    pub fn into_models(self) -> Vec<Model> {
        self.data.into_iter().map(Model::from).collect()
    }
}

impl From<ModelEntry> for Model {
    fn from(entry: ModelEntry) -> Self {
        let mut model = Model::from_id(entry.id);
        model.name = entry.name;
        model.owned_by = entry.vendor;
        if let Some(capabilities) = entry.capabilities {
            model.r#type = capabilities.kind;
        }
        model
    }
}

/// The model-listing decoder.
#[derive(Default)]
pub struct ModelsDecoder;

impl Decoder<ModelListing> for ModelsDecoder {
    type Event = ModelsReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<ModelListing>) {
        out.push(Ok(ModelList::new(event.into_models())));
    }
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let mut request = http::Request::get(self.provider.uri(super::MODEL_LISTING_PATH))
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::empty())?;
        stamp(
            &mut request,
            self.provider.api_key.expose(),
            "user",
            false,
            CopilotIntent::Panel,
        )?;
        Ok(Encoded::new(request, Framing::Whole).with_request_id_header(REQUEST_ID_HEADER))
    }

    fn decoder(&self, _mode: Mode) -> ModelsDecoder {
        ModelsDecoder
    }
}

impl HasCompletion for Copilot {
    type Wire = CopilotWire;

    fn completion(&self, model: impl Into<String>) -> CopilotWire {
        self.completion(model)
    }
}

impl HasEmbedding for Copilot {
    type Wire = Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        self.embeddings(model, ndims)
    }
}

impl HasModelListing for Copilot {
    type Wire = Models;

    fn model_listing(&self) -> Models {
        self.models()
    }
}

#[cfg(test)]
mod tests;
