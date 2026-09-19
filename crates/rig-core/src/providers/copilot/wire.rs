//! GitHub Copilot as data: one configuration, one completion wire that
//! chooses a route, and Copilot's own embeddings and catalogue endpoints.
//!
//! Copilot serves *two* completion APIs behind one host, and which one
//! answers is a property of the model: the Codex-class models take the
//! Responses request, everything else takes chat completions. The shared
//! [`OpenAiWire`] is already a wire over both routes, so [`CopilotWire`] is
//! that enum plus the one thing Copilot adds per turn — the conversation
//! intent — and every [`Wire`] method delegates to it. The choice itself is
//! made exactly once, in [`Copilot::completion`], by
//! [`routes_through_responses`]: per model, not per dialect, which is why
//! Copilot builds the variant itself rather than reading the dialect's
//! [`completion_route`](Quirks::completion_route).
//!
//! What is Copilot's own is the *envelope*: the editor identity every
//! request carries (`copilot-integration-id`, `editor-version`,
//! `openai-intent`, `X-Initiator`, …). Those are stamped onto the request
//! the delegated wire built, so there is one definition of the header set: a
//! crate-internal `default_headers` in the parent module, which every route
//! here stamps on.
//!
//! Copilot's session token is exchanged over the network before the API can
//! be called at all, and a pure synchronous [`Wire::encode`] has no seat for
//! a round trip. So the exchange stays in [`super::auth`] and this
//! configuration holds the credential it produced: [`Copilot::from_auth`] is
//! the bridge, and [`Copilot::new`] takes an already-exchanged token
//! directly.

use serde::{Deserialize, Serialize};

use crate::client::env::{self, EnvError};
use crate::completion::{CompletionError, CompletionRequest, ProviderCapabilities};
use crate::driver::{HasEmbedding, HasModelListing};
use crate::embeddings::EmbeddingError;
use crate::model::{Model, ModelList, ModelListingError};
use crate::operation::{Completion, Embedding, EmbeddingCapabilities, ModelListing};
use crate::providers::internal::wire::classify_untyped_line;
use crate::providers::openai::embedding::EncodingFormat;
use crate::providers::openai::responses_api::SystemInstructionsPlacement;
use crate::providers::openai::wire::{
    Dialect, EmbeddingQuirks, Embeddings as OpenAiEmbeddings,
    EmbeddingsDecoder as OpenAiEmbeddingsDecoder, OpenAI, OpenAiDecoder, OpenAiWire, Quirks,
    ResponsesQuirks,
};
use crate::telemetry::CompletionOperation;
use crate::wire::{
    Body, Decoder, Encoded, Framing, HasCompletion, Mode, Output, Secret, Sink, Wire, WireError,
    WireEvent, WireFrame,
};

use super::{CopilotIntent, PROVIDER_NAME};

/// The reply header carrying Copilot's transport request id. Copilot relays
/// OpenAI's wire on both routes, header included.
const REQUEST_ID_HEADER: Option<&str> = Some("x-request-id");

/// The credential, in order of precedence. Both spellings are documented, so
/// both are read.
/// The variable a missing-credential error names.
const PRIMARY_API_KEY_ENV: &str = "GITHUB_COPILOT_API_KEY";

/// The credential variables, in precedence order. A fixed-size array so the
/// first name — the one a missing-variable error reports — is reachable
/// without indexing.
const API_KEY_ENV: [&str; 2] = ["GITHUB_COPILOT_API_KEY", "COPILOT_API_KEY"];

/// The base-URL override, in order of precedence.
const BASE_URL_ENV: &[&str] = &["GITHUB_COPILOT_API_BASE", "COPILOT_BASE_URL"];

/// GitHub Copilot, as an OpenAI dialect.
///
/// Both routes are OpenAI's own contract — Copilot relays the chat and the
/// Responses wire verbatim, header included — except where the Responses
/// system preamble goes: this backend takes `system` messages inside
/// `input` rather than top-level `instructions`, which is what
/// `crates/rig-cassette/fixtures/cassettes/copilot/routing/codex_models_route_through_responses.yaml`
/// records. Copilot verifies through its token exchange, not a path, and
/// its editor headers and session-token exchange live in this module.
pub const DIALECT: Dialect = Dialect {
    base_url_env: Some("GITHUB_COPILOT_API_BASE"),
    request_id_header: REQUEST_ID_HEADER,
    quirks: Quirks {
        verify_path: "",
        base_url_env_alias: Some("COPILOT_BASE_URL"),
        embedding: EmbeddingQuirks {
            requires_usage: false,
            ..EmbeddingQuirks::openai()
        },
        responses: ResponsesQuirks {
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

/// Whether `model` is answered by Copilot's `/responses` route rather than
/// `/chat/completions`.
///
/// Copilot routes its Codex-class models through the Responses API and
/// everything else — the OpenAI chat models, and the Anthropic and Google
/// models it fronts — through chat completions. The predicate is the model
/// identifier alone, so it is a function and not a table: Copilot ships new
/// `*-codex` models without announcing them.
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
    /// Copilot, addressed with an already-exchanged session token.
    ///
    /// A session token names the REST endpoint it was minted for in its
    /// `proxy-ep=` segment, and nothing else knows it, so the base URL is
    /// derived from the token when it carries one. [`Self::with_base_url`]
    /// and the environment override it.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        let api_key = api_key.into();
        let base_url = super::base_url_from_token(api_key.expose())
            .unwrap_or_else(|| super::GITHUB_COPILOT_API_BASE_URL.to_owned());
        Self { api_key, base_url }
    }

    /// Copilot, addressed with the credential a token exchange resolved.
    ///
    /// The exchange itself is [`super::auth`]: it polls a device flow,
    /// refreshes a cached key and shares a token cache, none of which fits
    /// in a synchronous `encode`. Its [`AuthContext`](super::auth::AuthContext)
    /// also names the API base the credential belongs to, which takes
    /// precedence over the endpoint encoded in the token.
    pub fn from_auth(context: &super::auth::AuthContext) -> Self {
        let mut provider = Self::new(context.api_key.clone());
        if let Some(api_base) = &context.api_base {
            provider.base_url = api_base.clone();
        }
        provider
    }

    /// Copilot from the environment: `GITHUB_COPILOT_API_KEY` or
    /// `COPILOT_API_KEY` for the session token, `GITHUB_COPILOT_API_BASE` or
    /// `COPILOT_BASE_URL` for the base URL — the variables the client read.
    ///
    /// Only the already-exchanged credential is readable this way. A GitHub
    /// access token or a device-code login is a conversation, not a value:
    /// run [`super::auth`] and build the provider with [`Self::from_auth`].
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
        CopilotWire::from_openai(self.openai(), model)
    }

    /// The embeddings wire for `model`.
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        Embeddings::new(self.clone(), model, ndims)
    }

    /// The model-listing wire.
    pub fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }

    /// Copilot as the shared OpenAI configuration, which both completion
    /// routes are wires on.
    ///
    /// Built through the constructor rather than a struct literal, so a
    /// field the shared config grows for another dialect takes its own
    /// default here instead of having to be restated.
    fn openai(&self) -> OpenAI {
        OpenAI::with_key(&DIALECT, self.api_key.clone()).with_base_url(self.base_url.clone())
    }

    /// Resolve `path` against the base URL.
    fn uri(&self, path: &str) -> String {
        format!("{}{path}", self.base_url.trim_end_matches('/'))
    }
}

/// The first of `names` that is set to something other than blank.
///
/// A blank credential authenticates as nobody and a blank base URL addresses
/// nothing, so an empty value is "unset" here — the filter the client's
/// `env_value` applied.
fn first_env(names: &[&'static str]) -> Result<Option<String>, EnvError> {
    for name in names {
        if let Some(value) = env::optional(name)?.filter(|value| !value.trim().is_empty()) {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

/// Stamp Copilot's request envelope onto a request.
///
/// Applied to the finished request rather than threaded through each
/// encoder, because two of the three routes are shared wires this provider
/// does not own. `insert` replaces, so the `Authorization` a delegated wire
/// already set and the one here are the same header, not two.
fn stamp<E: WireError>(
    request: &mut http::Request<Body>,
    provider: &Copilot,
    initiator: &'static str,
    has_vision: bool,
    intent: CopilotIntent,
) -> Result<(), E> {
    let headers = super::default_headers(provider.api_key.expose(), initiator, has_vision, intent);
    let map = request.headers_mut();
    for (name, value) in &headers {
        let name = http::HeaderName::from_bytes(name.as_bytes())
            .map_err(|error| E::decode(error.to_string()))?;
        let value =
            http::HeaderValue::from_str(value).map_err(|error| E::decode(error.to_string()))?;
        map.insert(name, value);
    }
    Ok(())
}

// ── completion ──────────────────────────────────────────────────────────

/// Copilot's completion wire: whichever route this model is served by, plus
/// the conversation intent the turn declares.
///
/// The route is the shared [`OpenAiWire`] pointed at Copilot, so there is
/// no second request conversion, no second decoder and no second
/// observation projection for either API; what this type adds is the
/// `openai-intent` header and the rest of the editor envelope, stamped onto
/// the request the delegated wire built.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CopilotWire {
    /// The route's wire, pointed at Copilot.
    pub wire: OpenAiWire,
    /// The conversation intent this turn declares (`openai-intent`).
    pub intent: CopilotIntent,
}

impl CopilotWire {
    /// Keep an explicit OpenAI-shaped configuration intact while applying
    /// Copilot's model-dependent route and request envelope.
    pub(crate) fn from_openai(provider: OpenAI, model: impl Into<String>) -> Self {
        use crate::providers::openai::{Route, responses_api::wire::Responses, wire::Chat};

        let model = model.into();
        let route = provider.route.unwrap_or_else(|| {
            if routes_through_responses(&model) {
                Route::Responses
            } else {
                Route::Chat
            }
        });
        let wire = match route {
            // Copilot's Responses endpoint expects strict function schemas;
            // its Chat endpoint keeps strict mode opt-in.
            Route::Responses => {
                OpenAiWire::Responses(Responses::new(provider, model).with_strict_tools())
            }
            Route::Chat => OpenAiWire::Chat(Chat::new(provider, model)),
        };
        Self {
            wire,
            intent: CopilotIntent::default(),
        }
    }

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
    /// The Responses route already asks for it (see
    /// [`Copilot::completion`]), so this is the chat route's opt-in.
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

/// The Copilot credential behind a delegated wire's shared configuration.
///
/// Every route here but the catalogue is a wire pointed at Copilot through
/// [`Copilot::openai`], and [`stamp`] needs the credential back to build the
/// envelope: one direction, one definition.
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

    fn route(&self) -> Option<&str> {
        self.wire.route()
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        // Read off the request before the route's conversion consumes it —
        // the client layer's `RequestFacts::capture`, for the same reason.
        let initiator = super::request_initiator(&request);
        let has_vision = super::request_has_vision(&request);
        let mut encoded = self.wire.encode(request, mode)?;
        let provider = credential_of(self.wire.provider());
        for request in &mut encoded.requests {
            stamp::<CompletionError>(request, &provider, initiator, has_vision, self.intent)?;
        }
        Ok(encoded)
    }

    fn decoder(&self, mode: Mode) -> OpenAiDecoder {
        self.wire.decoder(mode)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.wire.capabilities()
    }

    fn telemetry(&self, streaming: bool) -> CompletionOperation {
        self.wire.telemetry(streaming)
    }
}

// ── embeddings ──────────────────────────────────────────────────────────

/// Copilot's embeddings wire: the shared embeddings wire pointed at Copilot,
/// plus the editor envelope.
///
/// Copilot relays OpenAI's embeddings contract verbatim — the path, the
/// width field, the `{ "data": [...] }` reply — and the one thing it is
/// measured to differ in is already [`DIALECT`]'s to state: the vendors it
/// fronts do not all report a usage block, hence
/// `embedding: EmbeddingQuirks { requires_usage: false, .. }`. So there is
/// no second request body, no second reply type and no second decoder here,
/// exactly as there is none for either completion route — only the `stamp`
/// every request to this host carries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The shared embeddings wire, pointed at Copilot.
    pub wire: OpenAiEmbeddings,
}

impl Embeddings {
    /// The embeddings wire for `model`.
    ///
    /// The width is the caller's, else the model's documented one: that
    /// resolution is the shared wire's, off the dialect's width table and
    /// OpenAI's `text-embedding-*` identifiers, so Copilot carries no second
    /// copy of the defaults.
    pub fn new(provider: Copilot, model: impl Into<String>, ndims: Option<usize>) -> Self {
        Self {
            wire: OpenAiEmbeddings::new(provider.openai(), model, ndims),
        }
    }

    /// Ask the provider to answer in `encoding_format`.
    pub fn with_encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.wire = self.wire.with_encoding_format(encoding_format);
        self
    }

    /// Attribute the call to an end user.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.wire = self.wire.with_user(user);
        self
    }
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Decoder = OpenAiEmbeddingsDecoder;

    fn name(&self) -> &str {
        self.wire.name()
    }

    fn model(&self) -> Option<&str> {
        self.wire.model()
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        self.wire.capabilities()
    }

    fn encode(&self, request: Vec<String>, mode: Mode) -> Result<Encoded, EmbeddingError> {
        let mut encoded = self.wire.encode(request, mode)?;
        let provider = credential_of(&self.wire.provider);
        for request in &mut encoded.requests {
            // The modality routes are not a conversation: the client layer
            // sent them the panel intent and a `user` initiator, and that is
            // what the recorded traffic carries.
            stamp::<EmbeddingError>(request, &provider, "user", false, CopilotIntent::Panel)?;
        }
        Ok(encoded)
    }

    fn decoder(&self, mode: Mode) -> OpenAiEmbeddingsDecoder {
        self.wire.decoder(mode)
    }
}

// ── model listing ───────────────────────────────────────────────────────

/// Copilot's model-listing wire.
///
/// `GET /models` answers with the whole catalogue, so
/// [`Decoder::continuation`] keeps its default `None`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// Which Copilot, and how to reach it.
    pub provider: Copilot,
}

/// One entry of Copilot's `{ "data": [...] }` catalogue.
///
/// Copilot names the vendor behind each model and nests the modality under
/// `capabilities.type`, which is where it differs from the OpenAI-shaped
/// listing every other dialect sends.
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

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, ModelListingError> {
        let mut request = http::Request::get(self.provider.uri(super::MODEL_LISTING_PATH))
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::empty())
            .map_err(|error| ModelListingError::RequestError {
                message: error.to_string(),
            })?;
        stamp::<ModelListingError>(
            &mut request,
            &self.provider,
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

// ── construction traits ─────────────────────────────────────────────────

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
