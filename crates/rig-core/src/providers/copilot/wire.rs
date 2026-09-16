//! GitHub Copilot as data: one configuration, one completion wire that
//! chooses a route, and Copilot's own embeddings and catalogue endpoints.
//!
//! Copilot serves *two* completion APIs behind one host, and which one
//! answers is a property of the model: the Codex-class models take the
//! Responses request, everything else takes chat completions. Both wire
//! formats already exist in this crate, so [`CopilotWire`] is an enum over
//! them — a wire choosing a wire, not a model — and every [`Wire`] method
//! delegates to the chosen one. The choice itself is made exactly once, in
//! [`Copilot::completion`], by [`routes_through_responses`].
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
use crate::embeddings::{self, EmbeddingError};
use crate::model::{Model, ModelList, ModelListingError};
use crate::operation::{Completion, Embedding, EmbeddingCapabilities, ModelListing};
use crate::providers::internal::wire::classify_untyped_line;
use crate::providers::openai::completion::Usage;
use crate::providers::openai::embedding::EncodingFormat;
use crate::providers::openai::responses_api::{self, SystemInstructionsPlacement};
use crate::providers::openai::wire as chat_wire;
use crate::telemetry::CompletionOperation;
use crate::wire::{
    Body, Decoder, Encoded, Framing, HasCompletion, Mode, ObservationSink, Output, Secret, Sink,
    Wire, WireError, WireEvent, WireFrame,
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

/// Copilot's `/responses` route, as a Responses dialect.
///
/// Everything but the identity is OpenAI's own contract — Copilot relays the
/// Responses wire verbatim — except where the system preamble goes: this
/// backend takes `system` messages inside `input` rather than top-level
/// `instructions`, which is what
/// `tests/cassettes/copilot/routing/codex_models_route_through_responses.yaml`
/// records.
pub const DIALECT: responses_api::wire::Dialect = responses_api::wire::Dialect {
    name: PROVIDER_NAME,
    base_url: super::GITHUB_COPILOT_API_BASE_URL,
    api_key_env: "GITHUB_COPILOT_API_KEY",
    base_url_env: Some("GITHUB_COPILOT_API_BASE"),
    request_id_header: Some("x-request-id"),
    quirks: responses_api::wire::Quirks {
        path: "/responses",
        system_instructions: SystemInstructionsPlacement::InputSystemMessages,
        request: responses_api::wire::RequestShape::Responses,
        base_url_env_alias: Some("COPILOT_BASE_URL"),
        account_id_env: None,
        default_instructions: None,
        instructions_env: None,
        identity: None,
        always_streams: false,
        relaxed_content_type: false,
        codex_parameter_subset: false,
        error_envelope_in_success: false,
        repair_envelope_less_frames: false,
        native_output_with_tools: true,
    },
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
        let mut provider = Self::new(context.api_key.as_str());
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

    /// The one place the route is chosen. Named apart from the two public
    /// entry points ([`Self::completion`] and [`HasCompletion::completion`])
    /// so neither can delegate to the other by accident.
    fn wire_for(&self, model: impl Into<String>) -> CopilotWire {
        let model = model.into();
        let intent = CopilotIntent::default();
        if routes_through_responses(&model) {
            // Copilot's Responses route wants strict function schemas for
            // reliable tool calls; the chat route keeps strict mode opt-in,
            // exactly as the client layer had it.
            let wire = self.responses_api().responses(model).with_strict_tools();
            CopilotWire::Responses { wire, intent }
        } else {
            CopilotWire::Chat {
                wire: self.chat_api().chat(model),
                intent,
            }
        }
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

    /// Copilot's chat route, as an OpenAI-shaped configuration.
    ///
    /// Built through the constructors rather than a struct literal, so a
    /// field the shared config grows for another dialect takes its own
    /// default here instead of having to be restated.
    fn chat_api(&self) -> chat_wire::OpenAI {
        chat_wire::OpenAI::with_key(&chat_wire::COPILOT_CHAT, self.api_key.clone())
            .with_base_url(self.base_url.clone())
    }

    /// Copilot's Responses route, as a Responses-format configuration.
    fn responses_api(&self) -> responses_api::wire::ResponsesApi {
        responses_api::wire::ResponsesApi::with_dialect(self.api_key.clone(), &DIALECT)
            .with_base_url(self.base_url.clone())
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

/// Copilot's completion wire: whichever route this model is served by.
///
/// A wire choosing a wire. Both variants are shared wire types pointed at
/// Copilot, and every [`Wire`] method dispatches on the variant, so there is
/// no second request conversion, no second decoder and no second observation
/// projection for either API.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CopilotWire {
    /// The conversational models: `POST /chat/completions`.
    Chat {
        /// The shared chat-completions wire, pointed at Copilot.
        wire: chat_wire::Chat,
        /// The conversation intent this turn declares (`openai-intent`).
        intent: CopilotIntent,
    },
    /// The Codex-class models: `POST /responses`.
    Responses {
        /// The shared Responses wire, pointed at Copilot.
        wire: responses_api::wire::Responses,
        /// The conversation intent this turn declares (`openai-intent`).
        intent: CopilotIntent,
    },
}

impl CopilotWire {
    /// The conversation intent this wire declares.
    pub fn intent(&self) -> CopilotIntent {
        match self {
            Self::Chat { intent, .. } | Self::Responses { intent, .. } => *intent,
        }
    }

    /// Declare `intent` in the `openai-intent` header.
    pub fn with_intent(self, intent: CopilotIntent) -> Self {
        match self {
            Self::Chat { wire, .. } => Self::Chat { wire, intent },
            Self::Responses { wire, .. } => Self::Responses { wire, intent },
        }
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
    pub fn with_strict_tools(self) -> Self {
        match self {
            Self::Chat { wire, intent } => Self::Chat {
                wire: wire.with_strict_tools(),
                intent,
            },
            Self::Responses { wire, intent } => Self::Responses {
                wire: wire.with_strict_tools(),
                intent,
            },
        }
    }

    /// Serialize tool-result content as arrays.
    ///
    /// A chat-completions shape: the Responses request has one content
    /// encoding, so this is a no-op on that route.
    pub fn with_tool_result_array_content(self) -> Self {
        match self {
            Self::Chat { wire, intent } => Self::Chat {
                wire: wire.with_tool_result_array_content(),
                intent,
            },
            responses @ Self::Responses { .. } => responses,
        }
    }

    /// The credential this wire sends, for the request envelope.
    fn provider(&self) -> Copilot {
        let (api_key, base_url) = match self {
            Self::Chat { wire, .. } => (&wire.provider.api_key, &wire.provider.base_url),
            Self::Responses { wire, .. } => (&wire.provider.api_key, &wire.provider.base_url),
        };
        Copilot {
            api_key: api_key.clone(),
            base_url: base_url.clone(),
        }
    }
}

impl Wire for CopilotWire {
    type Op = Completion;
    type Decoder = CopilotDecoder;

    fn name(&self) -> &str {
        match self {
            Self::Chat { wire, .. } => wire.name(),
            Self::Responses { wire, .. } => wire.name(),
        }
    }

    fn model(&self) -> Option<&str> {
        match self {
            Self::Chat { wire, .. } => wire.model(),
            Self::Responses { wire, .. } => wire.model(),
        }
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        // Read off the request before the route's conversion consumes it —
        // the client layer's `RequestFacts::capture`, for the same reason.
        let initiator = super::request_initiator(&request);
        let has_vision = super::request_has_vision(&request);
        let intent = self.intent();
        let mut encoded = match self {
            Self::Chat { wire, .. } => wire.encode(request, mode)?,
            Self::Responses { wire, .. } => wire.encode(request, mode)?,
        };
        let provider = self.provider();
        for request in &mut encoded.requests {
            stamp::<CompletionError>(request, &provider, initiator, has_vision, intent)?;
        }
        Ok(encoded)
    }

    fn decoder(&self) -> CopilotDecoder {
        match self {
            Self::Chat { wire, .. } => CopilotDecoder::Chat(wire.decoder()),
            Self::Responses { wire, .. } => CopilotDecoder::Responses(wire.decoder()),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        match self {
            Self::Chat { wire, .. } => wire.capabilities(),
            Self::Responses { wire, .. } => wire.capabilities(),
        }
    }

    fn telemetry(&self, streaming: bool) -> CompletionOperation {
        match self {
            Self::Chat { wire, .. } => wire.telemetry(streaming),
            Self::Responses { wire, .. } => wire.telemetry(streaming),
        }
    }
}

/// One classified frame of whichever Copilot route is answering.
pub enum CopilotEvent {
    /// A chat-completions frame.
    Chat(chat_wire::ChatEvent),
    /// A Responses frame.
    Responses(responses_api::streaming::ResponsesEvent),
}

/// Copilot's completion decoder: the chosen route's decoder.
pub enum CopilotDecoder {
    /// The chat-completions state machine.
    Chat(chat_wire::ChatDecoder),
    /// The Responses state machine.
    Responses(responses_api::streaming::ResponsesDecoder),
}

impl Decoder<Completion> for CopilotDecoder {
    type Event = CopilotEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        match self {
            Self::Chat(decoder) => decoder.classify(frame).map(CopilotEvent::Chat),
            Self::Responses(decoder) => decoder.classify(frame).map(CopilotEvent::Responses),
        }
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Completion>) {
        match (self, event) {
            (Self::Chat(decoder), CopilotEvent::Chat(event)) => decoder.interpret(event, out),
            (Self::Responses(decoder), CopilotEvent::Responses(event)) => {
                decoder.interpret(event, out);
            }
            // A decoder is built fresh per reply and only ever sees the
            // events its own `classify` produced, so the routes cannot
            // cross; there is nothing for the other route's state machine to
            // do with a frame it never decoded.
            (Self::Chat(_), CopilotEvent::Responses(_))
            | (Self::Responses(_), CopilotEvent::Chat(_)) => {}
        }
    }

    fn finish(&mut self, out: &mut Output<Completion>) {
        match self {
            Self::Chat(decoder) => decoder.finish(out),
            Self::Responses(decoder) => decoder.finish(out),
        }
    }

    fn flush_before_terminal_error(&mut self, out: &mut Output<Completion>) {
        match self {
            Self::Chat(decoder) => decoder.flush_before_terminal_error(out),
            Self::Responses(decoder) => decoder.flush_before_terminal_error(out),
        }
    }

    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
        match self {
            Self::Chat(decoder) => decoder.project(payload, sink),
            Self::Responses(decoder) => decoder.project(payload, sink),
        }
    }

    fn continuation(&self) -> Option<http::Request<Body>> {
        match self {
            Self::Chat(decoder) => decoder.continuation(),
            Self::Responses(decoder) => decoder.continuation(),
        }
    }

    fn is_analysis_only(&self, frame: &WireFrame) -> bool {
        match self {
            Self::Chat(decoder) => decoder.is_analysis_only(frame),
            Self::Responses(decoder) => decoder.is_analysis_only(frame),
        }
    }

    fn is_finished(&self) -> bool {
        match self {
            Self::Chat(decoder) => decoder.is_finished(),
            Self::Responses(decoder) => decoder.is_finished(),
        }
    }
}

// ── embeddings ──────────────────────────────────────────────────────────

/// Copilot's embeddings path.
const EMBEDDINGS_PATH: &str = "/embeddings";

/// The most inputs Copilot embeds in one request.
const MAX_DOCUMENTS: usize = 1024;

/// Copilot's embeddings wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// Which Copilot, and how to reach it.
    pub provider: Copilot,
    /// The embedding model.
    pub model: String,
    /// The vector width, resolved at construction from the model identifier
    /// when the caller named none. Zero means this build knows no width for
    /// the model, and none is sent.
    pub ndims: usize,
    /// The encoding the caller asked the provider to answer in.
    pub encoding_format: Option<EncodingFormat>,
    /// The end-user identifier the provider attributes the call to.
    pub user: Option<String>,
}

impl Embeddings {
    /// The embeddings wire for `model`, defaulting the width from the model
    /// identifier when the caller gave none.
    pub fn new(provider: Copilot, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let ndims = ndims.unwrap_or(match model.as_str() {
            super::TEXT_EMBEDDING_3_LARGE => 3072,
            super::TEXT_EMBEDDING_3_SMALL | super::TEXT_EMBEDDING_ADA_002 => 1536,
            _ => 0,
        });
        Self {
            provider,
            model,
            ndims,
            encoding_format: None,
            user: None,
        }
    }

    /// Ask the provider to answer in `encoding_format`.
    pub fn with_encoding_format(mut self, encoding_format: EncodingFormat) -> Self {
        self.encoding_format = Some(encoding_format);
        self
    }

    /// Attribute the call to an end user.
    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }
}

/// Copilot's embeddings reply.
///
/// Copilot fronts several vendors, so neither the usage block nor the model
/// identifier is guaranteed on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsReply {
    /// One datum per input, in input order.
    pub data: Vec<EmbeddingDatum>,
    /// Token usage, when the answering vendor reported any.
    #[serde(default)]
    pub usage: Option<Usage>,
    /// The model that answered, when named.
    #[serde(default)]
    pub model: Option<String>,
}

/// One embedded input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDatum {
    /// The vector, as the provider sent it.
    pub embedding: Vec<serde_json::Number>,
}

/// The embeddings decoder.
#[derive(Default)]
pub struct EmbeddingsDecoder;

impl Decoder<Embedding> for EmbeddingsDecoder {
    type Event = EmbeddingsReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Embedding>) {
        // A missing usage payload reports no counter at all rather than
        // failing the call: Copilot's multi-vendor route omits it.
        let usage = event
            .usage
            .as_ref()
            .map(Usage::to_normalized)
            .unwrap_or_default();
        let embeddings = event
            .data
            .into_iter()
            .map(|datum| embeddings::Embedding {
                // Joined back onto the request's inputs by the operation's
                // fold, which is the only place that still holds them.
                document: String::new(),
                vec: datum
                    .embedding
                    .into_iter()
                    .filter_map(|number| number.as_f64())
                    .collect(),
            })
            .collect();
        out.push(Ok(embeddings::EmbeddingResponse::new(
            embeddings,
            PROVIDER_NAME,
        )
        .with_optional_model(event.model)
        .with_usage(usage)));
    }
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Decoder = EmbeddingsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(MAX_DOCUMENTS, self.ndims)
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EmbeddingError> {
        let mut body = serde_json::json!({ "model": self.model, "input": request });
        let Some(object) = body.as_object_mut() else {
            return Err(EmbeddingError::ResponseError(
                "embedding request body must be a JSON object".into(),
            ));
        };
        // The legacy Ada model takes no width, and zero means this build
        // knows none; neither is a value to send.
        if self.ndims > 0 && self.model != super::TEXT_EMBEDDING_ADA_002 {
            object.insert("dimensions".to_owned(), serde_json::json!(self.ndims));
        }
        if let Some(encoding_format) = self.encoding_format {
            object.insert(
                "encoding_format".to_owned(),
                serde_json::to_value(encoding_format)?,
            );
        }
        if let Some(user) = &self.user {
            object.insert("user".to_owned(), serde_json::json!(user));
        }

        let mut request = http::Request::post(self.provider.uri(EMBEDDINGS_PATH))
            .header(http::header::CONTENT_TYPE, "application/json")
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|error| EmbeddingError::ResponseError(error.to_string()))?;
        // The modality routes are not a conversation: the client layer sent
        // them the panel intent and a `user` initiator, and that is what the
        // recorded traffic carries.
        stamp::<EmbeddingError>(
            &mut request,
            &self.provider,
            "user",
            false,
            CopilotIntent::Panel,
        )?;
        Ok(Encoded::new(request, Framing::Whole).with_request_id_header(REQUEST_ID_HEADER))
    }

    fn decoder(&self) -> EmbeddingsDecoder {
        EmbeddingsDecoder
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

    fn decoder(&self) -> ModelsDecoder {
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
