//! The OpenAI wires that are not chat: embeddings, transcription, images,
//! speech, model listing and credential verification.
//!
//! Each is one request and one reply document, so each decoder is one
//! `classify` that delegates to the untyped-line classifier and one
//! `interpret` that pushes the single event its operation folds.

use serde::{Deserialize, Serialize};

#[cfg(feature = "audio")]
use crate::audio_generation::AudioGenerationError;
use crate::client::VerifyError;
use crate::embeddings::{self, EmbeddingError};
use crate::model::{Model, ModelList, ModelListingError};
use crate::rerank::RerankError;
use crate::operation::{
    Embedding, EmbeddingCapabilities, ModelListing, Rerank as RerankOp, Transcription,
    Verify as VerifyOp,
};
use crate::providers::internal::wire::classify_untyped_line;
use crate::providers::openai::completion::Usage;
use crate::providers::openai::embedding::{EncodingFormat, model_dimensions_from_identifier};
use crate::transcription::{TranscriptionError, TranscriptionRequest};
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame,
};

use super::{DimensionsField, OpenAI, Routing};
// Each is read by exactly one feature-gated wire.
#[cfg(feature = "image")]
use super::ImageBody;
#[cfg(feature = "audio")]
use super::SpeechBody;

/// Build the JSON request this dialect's endpoint expects.
///
/// Generic over the operation's error so a base URL that does not parse as a
/// URI fails the operation it belongs to, rather than being reported against
/// some other one.
fn json_request<E: crate::wire::WireError>(
    provider: &OpenAI,
    path: &str,
    deployment: Option<&str>,
    body: &serde_json::Value,
) -> Result<http::Request<Body>, E> {
    json_request_to(provider, provider.uri(path, deployment), body)
}

/// [`json_request`] against an already-resolved URL, for the endpoints whose
/// URL is derived rather than a fixed path under the base.
fn json_request_to<E: crate::wire::WireError>(
    provider: &OpenAI,
    uri: String,
    body: &serde_json::Value,
) -> Result<http::Request<Body>, E> {
    let bytes = serde_json::to_vec(body).map_err(E::json)?;
    let builder = http::Request::post(uri).header("Content-Type", "application/json");
    provider
        .authenticate(builder)
        .body(Body::Bytes(bytes))
        .map_err(|error| E::decode(error.to_string()))
}

/// The deployment segment Azure routes a model through, or `None`.
fn deployment<'a>(provider: &OpenAI, model: &'a str) -> Option<&'a str> {
    match provider.dialect.quirks.routing {
        Routing::AzureDeployment => Some(model),
        Routing::Path => None,
    }
}

// ── embeddings ──────────────────────────────────────────────────────────

/// The embeddings wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
    /// The embedding model.
    pub model: String,
    /// The width the caller asked for, when they named one rather than
    /// taking the model's default.
    pub ndims: Option<usize>,
    /// The encoding the caller asked the provider to answer in.
    pub encoding_format: Option<EncodingFormat>,
    /// The end-user identifier the provider attributes the call to.
    pub user: Option<String>,
}

impl Embeddings {
    /// The embeddings wire for `model`.
    pub fn new(provider: OpenAI, model: impl Into<String>, ndims: Option<usize>) -> Self {
        Self {
            provider,
            model: model.into(),
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

    /// The width this wire reports, which is the caller's when they named
    /// one and the model's documented width otherwise. Zero means unknown —
    /// the model is absent from every table this build knows.
    fn resolved_ndims(&self) -> usize {
        self.ndims
            .or_else(|| model_dimensions_from_identifier(&self.model))
            .unwrap_or_default()
    }

    /// The width to send, in the field this dialect spells it with.
    ///
    /// OpenAI's legacy Ada model does not accept a width at all, and
    /// `llama-server` reads no width field, so neither is sent one.
    fn requested_width(&self) -> Option<(&'static str, usize)> {
        let ndims = self.ndims?;
        match self.provider.dialect.quirks.embedding.dimensions {
            DimensionsField::Ignored => None,
            _ if self.model == crate::providers::openai::embedding::TEXT_EMBEDDING_ADA_002 => None,
            DimensionsField::Dimensions => Some(("dimensions", ndims)),
            DimensionsField::OutputDimension => Some(("output_dimension", ndims)),
        }
    }
}

/// The embeddings reply, as every dialect on this wire answers it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsReply {
    #[serde(default)]
    pub object: String,
    pub data: Vec<EmbeddingDatum>,
    #[serde(default)]
    pub model: String,
    /// Optional because compatible dialects may omit it.
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDatum {
    #[serde(default)]
    pub object: String,
    pub embedding: Vec<serde_json::Number>,
    #[serde(default)]
    pub index: usize,
}

/// The embeddings decoder.
#[derive(Default)]
pub struct EmbeddingsDecoder {
    /// Whether this dialect's reply must carry usage.
    requires_usage: bool,
    provider: &'static str,
    model: String,
}

impl Decoder<Embedding> for EmbeddingsDecoder {
    type Event = EmbeddingsReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Embedding>) {
        if event.usage.is_none() && self.requires_usage {
            out.push(Err(EmbeddingError::MissingUsage {
                provider: self.provider,
            }));
            return;
        }
        let usage = event
            .usage
            .as_ref()
            .map(Usage::to_normalized)
            .unwrap_or_default();
        // `document` is joined on by the operation's fold, which is the only
        // place that still holds the request's inputs.
        let embeddings = event
            .data
            .into_iter()
            .map(|datum| embeddings::Embedding {
                document: String::new(),
                vec: datum
                    .embedding
                    .into_iter()
                    .filter_map(|number| number.as_f64())
                    .collect(),
            })
            .collect();
        let model = if event.model.is_empty() {
            self.model.clone()
        } else {
            event.model
        };
        out.push(Ok(embeddings::EmbeddingResponse::new(
            embeddings,
            self.provider,
        )
        .with_model(model)
        .with_usage(usage)));
    }
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Decoder = EmbeddingsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(
            self.provider.dialect.quirks.embedding.max_documents,
            self.resolved_ndims(),
        )
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EmbeddingError> {
        let quirks = &self.provider.dialect.quirks.embedding;
        // Base64 vectors are not decoded anywhere, so asking for them would
        // answer 200 with a payload rig cannot read.
        if self.encoding_format == Some(EncodingFormat::Base64) {
            return Err(EmbeddingError::UnsupportedResponseEncoding {
                provider: self.provider.dialect.name,
                encoding_format: "base64",
            });
        }
        if self.encoding_format.is_some() && !quirks.supports_encoding_format {
            return Err(EmbeddingError::UnsupportedParameter {
                provider: self.provider.dialect.name,
                parameter: "encoding_format",
            });
        }
        if self.user.is_some() && !quirks.supports_user {
            return Err(EmbeddingError::UnsupportedParameter {
                provider: self.provider.dialect.name,
                parameter: "user",
            });
        }

        let mut body = serde_json::json!({ "input": request });
        let Some(object) = body.as_object_mut() else {
            return Err(EmbeddingError::ResponseError(
                "embedding request body must be an object".into(),
            ));
        };
        if quirks.sends_model_field {
            object.insert("model".to_owned(), serde_json::json!(self.model));
        }
        if let Some((field, ndims)) = self.requested_width() {
            object.insert(field.to_owned(), serde_json::json!(ndims));
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

        let request = json_request::<EmbeddingError>(
            &self.provider,
            self.provider.dialect.quirks.embeddings_path,
            deployment(&self.provider, &self.model),
            &body,
        )?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> EmbeddingsDecoder {
        EmbeddingsDecoder {
            requires_usage: self.provider.dialect.quirks.embedding.requires_usage,
            provider: self.provider.dialect.name,
            model: self.model.clone(),
        }
    }
}

// ── transcription ───────────────────────────────────────────────────────

/// The transcription wire: the only one whose body is multipart.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transcriptions {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
    /// The transcription model, or — for Azure — the deployment.
    pub model: String,
}

impl Transcriptions {
    /// The transcription wire for `model`.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

/// The transcription decoder.
#[derive(Default)]
pub struct TranscriptionsDecoder {
    provider: &'static str,
}

impl Decoder<Transcription> for TranscriptionsDecoder {
    type Event = crate::providers::openai::transcription::TranscriptionResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Transcription>) {
        use crate::transcription::NormalizeTranscriptionResponse;

        match serde_json::to_value(&event) {
            Ok(raw) => match event.normalize(self.provider) {
                Ok(response) => out.push(Ok(response.with_raw(raw))),
                Err(error) => out.push(Err(error)),
            },
            Err(error) => out.push(Err(TranscriptionError::from(error))),
        }
    }
}

impl Wire for Transcriptions {
    type Op = Transcription;
    type Decoder = TranscriptionsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(
        &self,
        request: TranscriptionRequest,
        _mode: Mode,
    ) -> Result<Encoded, TranscriptionError> {
        use crate::http_client::multipart::Part;
        use crate::http_client::MultipartForm;

        let mut form = MultipartForm::new();
        // Azure addresses a deployment in the URL and sends no model field;
        // every other dialect names the model in the form. Field order
        // matches the order these endpoints were built by hand, so recorded
        // requests stay byte-comparable.
        if !matches!(self.provider.dialect.quirks.routing, Routing::AzureDeployment) {
            form = form.text("model", self.model.clone());
        }
        form = form.part(Part::bytes("file", request.data).filename(request.filename));
        if let Some(language) = request.language {
            form = form.text("language", language);
        }
        if let Some(prompt) = request.prompt {
            form = form.text("prompt", prompt);
        }
        if let Some(temperature) = request.temperature {
            form = form.text("temperature", temperature.to_string());
        }
        if let Some(additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional transcription parameters must be a JSON object",
                )))
            })?;
            for (name, value) in params {
                // String values go on the form verbatim — `Value::to_string`
                // would send them JSON-quoted (`"verbose_json"`), which
                // providers reject or ignore. Non-string values stay JSON.
                let value = match value {
                    serde_json::Value::String(value) => value.clone(),
                    other => other.to_string(),
                };
                form = form.text(name.clone(), value);
            }
        }

        let uri = self
            .provider
            .modality_uri(
                "transcription",
                self.provider.dialect.quirks.transcription_path,
                &self.model,
            )
            .map_err(TranscriptionError::ProviderError)?;
        let builder = http::Request::post(uri);
        let request = self
            .provider
            .authenticate(builder)
            .body(Body::Multipart(form))
            .map_err(|error| TranscriptionError::ResponseError(error.to_string()))?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> TranscriptionsDecoder {
        TranscriptionsDecoder {
            provider: self.provider.dialect.name,
        }
    }
}

// ── image generation ────────────────────────────────────────────────────

/// The image-generation wire.
#[cfg(feature = "image")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Images {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
    /// The image model.
    pub model: String,
}

#[cfg(feature = "image")]
impl Images {
    /// The image-generation wire for `model`.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

/// The image-generation decoder.
#[cfg(feature = "image")]
#[derive(Default)]
pub struct ImagesDecoder {
    provider: &'static str,
}

/// One generated image, as every dialect on this wire returns it.
#[cfg(feature = "image")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDatum {
    /// The image, base64-encoded.
    pub b64_json: String,
}

/// The image-generation reply.
///
/// `created` is optional and the rest of the object is kept verbatim: OpenAI
/// sends `{created, data}` and xAI sends `{data}` alone, so a required
/// `created` would fail every xAI reply.
#[cfg(feature = "image")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesReply {
    /// The generated images.
    #[serde(default)]
    pub data: Vec<ImageDatum>,
    /// Whatever else the dialect sent (`created`, and any field this build
    /// does not model), so the raw payload loses nothing.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[cfg(feature = "image")]
impl Decoder<crate::operation::ImageGeneration> for ImagesDecoder {
    type Event = ImagesReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(
        &mut self,
        event: Self::Event,
        out: &mut Output<crate::operation::ImageGeneration>,
    ) {
        use base64::Engine;
        use crate::image_generation::{ImageGenerationError, ImageGenerationResponse};

        let Some(encoded) = event.data.first().map(|image| image.b64_json.as_str()) else {
            out.push(Err(ImageGenerationError::ResponseError(
                "missing image data".to_owned(),
            )));
            return;
        };
        let image = match base64::prelude::BASE64_STANDARD.decode(encoded) {
            Ok(image) => image,
            Err(error) => {
                out.push(Err(ImageGenerationError::ResponseError(error.to_string())));
                return;
            }
        };
        let raw = serde_json::to_value(&event).unwrap_or(serde_json::Value::Null);
        out.push(Ok(
            ImageGenerationResponse::new(image, self.provider).with_raw(raw)
        ));
    }
}

#[cfg(feature = "image")]
impl Wire for Images {
    type Op = crate::operation::ImageGeneration;
    type Decoder = ImagesDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(
        &self,
        request: crate::image_generation::ImageGenerationRequest,
        _mode: Mode,
    ) -> Result<Encoded, crate::image_generation::ImageGenerationError> {
        let mut body = match self.provider.dialect.quirks.image_body {
            // `response_format` is deliberately absent: it is no longer part
            // of OpenAI's request schema, which rejects it before it even
            // looks at the model. A compatible endpoint that still takes the
            // field gets it through `additional_params`.
            ImageBody::OpenAi => serde_json::json!({
                "model": self.model,
                "prompt": request.prompt,
                "size": format!("{}x{}", request.width, request.height),
            }),
            // xAI takes no `size` and answers with a URL unless asked for
            // base64, which is the only form this wire decodes.
            ImageBody::Xai => serde_json::json!({
                "model": self.model,
                "prompt": request.prompt,
                "response_format": "b64_json",
                "aspect_ratio": "1:1",
            }),
        };
        // Merged last, so a caller can reach the endpoint's other parameters
        // and override what is derived above.
        if let Some(additional_params) = request.additional_params {
            crate::json_utils::merge_inplace(&mut body, additional_params);
        }

        let uri = self
            .provider
            .modality_uri(
                "image generation",
                self.provider.dialect.quirks.image_generation_path,
                &self.model,
            )
            .map_err(crate::image_generation::ImageGenerationError::ProviderError)?;
        let request = json_request_to::<crate::image_generation::ImageGenerationError>(
            &self.provider,
            uri,
            &body,
        )?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> ImagesDecoder {
        ImagesDecoder {
            provider: self.provider.dialect.name,
        }
    }
}

// ── speech ──────────────────────────────────────────────────────────────

/// The speech wire.
#[cfg(feature = "audio")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Speech {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
    /// The speech model.
    pub model: String,
}

#[cfg(feature = "audio")]
impl Speech {
    /// The speech wire for `model`.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

/// The speech decoder.
///
/// This endpoint answers with the audio bytes themselves and no JSON
/// envelope, so the frame *is* the payload.
#[cfg(feature = "audio")]
#[derive(Default)]
pub struct SpeechDecoder {
    provider: &'static str,
}

#[cfg(feature = "audio")]
impl Decoder<crate::operation::AudioGeneration> for SpeechDecoder {
    type Event = Vec<u8>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        // Not JSON at all: there is nothing to decode, so there is nothing
        // to classify — the bytes are the answer.
        WireEvent::Known(match frame {
            WireFrame::Text(text) => text.into_bytes(),
            WireFrame::Bytes(bytes) => bytes,
        })
    }

    fn interpret(
        &mut self,
        event: Self::Event,
        out: &mut Output<crate::operation::AudioGeneration>,
    ) {
        out.push(Ok(crate::audio_generation::AudioGenerationResponse::new(
            event,
            self.provider,
        )));
    }
}

#[cfg(feature = "audio")]
impl Wire for Speech {
    type Op = crate::operation::AudioGeneration;
    type Decoder = SpeechDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(
        &self,
        request: crate::audio_generation::AudioGenerationRequest,
        _mode: Mode,
    ) -> Result<Encoded, AudioGenerationError> {
        let mut body = match self.provider.dialect.quirks.speech_body {
            SpeechBody::OpenAi => serde_json::json!({
                "model": self.model,
                "input": request.text,
                "voice": request.voice,
                "speed": request.speed,
            }),
            // xAI's `/v1/tts` names the voice `voice_id` and takes no model
            // in the body; `eve` is the voice its client defaulted to.
            SpeechBody::Xai => serde_json::json!({
                "text": request.text,
                "voice_id": if request.voice.is_empty() { "eve" } else { request.voice.as_str() },
                "language": "en",
            }),
        };
        // Last, so a caller can reach the endpoint's other parameters —
        // `response_format`, `instructions` — and override what is derived
        // above. They demonstrably change the reply: `response_format: "wav"`
        // returns RIFF where the default returns MP3.
        if let Some(additional_params) = request.additional_params {
            crate::json_utils::merge_inplace(&mut body, additional_params);
        }

        // Azure versions its speech endpoint separately from every other
        // route, so this one request carries its own `api-version`.
        let uri = self.provider.uri_versioned(
            self.provider.dialect.quirks.audio_generation_path,
            deployment(&self.provider, &self.model),
            self.provider.speech_api_version(),
        );
        let request = json_request_to::<AudioGenerationError>(&self.provider, uri, &body)?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> SpeechDecoder {
        SpeechDecoder {
            provider: self.provider.dialect.name,
        }
    }
}

// ── model listing ───────────────────────────────────────────────────────

/// The model-listing wire.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
}

impl Models {
    /// The model-listing wire.
    pub fn new(provider: OpenAI) -> Self {
        Self { provider }
    }
}

/// One entry of an OpenAI-style `{ "data": [...] }` listing.
///
/// `id` is the one field every dialect on this wire sends; the rest are
/// optional so a dialect that omits them still decodes. Groq additionally
/// reports the context window and output cap, which map onto [`Model`]'s own
/// fields rather than being dropped.
#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
}

/// The `{ "data": [...] }` envelope.
#[derive(Debug, Deserialize)]
pub struct ModelsReply {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

/// The model-listing decoder.
///
/// This endpoint is not paged — it answers with the whole catalogue — so
/// [`Decoder::continuation`] keeps its default `None`.
#[derive(Default)]
pub struct ModelsDecoder;

impl Decoder<ModelListing> for ModelsDecoder {
    type Event = ModelsReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<ModelListing>) {
        let models = event
            .data
            .into_iter()
            .map(|entry| {
                let mut model = Model::from_id(entry.id);
                model.name = entry.name;
                model.created_at = entry.created;
                model.owned_by = entry.owned_by;
                model.context_length = entry.context_window;
                model.max_output_tokens = entry.max_completion_tokens;
                model
            })
            .collect();
        out.push(Ok(ModelList::new(models)));
    }
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, ModelListingError> {
        let builder = http::Request::get(
            self.provider
                .uri(self.provider.dialect.quirks.models_path, None),
        );
        let request = self
            .provider
            .authenticate(builder)
            .body(Body::empty())
            .map_err(|error| ModelListingError::RequestError {
                message: error.to_string(),
            })?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> ModelsDecoder {
        ModelsDecoder
    }
}

// ── reranking ───────────────────────────────────────────────────────────

/// The rerank wire.
///
/// There is no reranking endpoint in the OpenAI API, so the compatible
/// servers that offer one converged on Jina's shape:
/// `{model, query, documents, top_n}` answered with
/// `{model, results:[{index, relevance_score}], usage}`. `llama-server`
/// serves exactly that.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Rerank {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
    /// The reranker model.
    pub model: String,
    /// Return only the `top_n` highest-scoring documents, when the caller
    /// asked for a cut.
    pub top_n: Option<usize>,
}

impl Rerank {
    /// The rerank wire for `model`.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
            top_n: None,
        }
    }

    /// Ask the server to return only the `top_n` highest-scoring documents.
    pub fn with_top_n(mut self, top_n: usize) -> Self {
        self.top_n = Some(top_n);
        self
    }
}

/// One scored document.
///
/// The score key is `relevance_score` on the Jina-shaped path and `score` on
/// the text-embeddings-inference path the same llama.cpp handler switches to;
/// both are accepted, so a server answering either shape decodes rather than
/// silently scoring every document zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankResultEntry {
    /// Which input document this scored.
    pub index: usize,
    /// The score.
    #[serde(alias = "score")]
    pub relevance_score: f64,
    /// Present only on servers that echo the document back; llama.cpp does
    /// not on this path.
    #[serde(default, alias = "text")]
    pub document: Option<String>,
}

/// What a rerank reply reports besides its ranking.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RerankUsage {
    /// Tokens the query and documents cost.
    #[serde(default)]
    pub prompt_tokens: u64,
    /// Total tokens, as the provider reported them.
    #[serde(default)]
    pub total_tokens: u64,
}

/// The rerank reply.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankReply {
    /// The model the server ranked with, when it named one.
    #[serde(default)]
    pub model: Option<String>,
    /// The ranking.
    pub results: Vec<RerankResultEntry>,
    /// What it cost.
    #[serde(default)]
    pub usage: Option<RerankUsage>,
}

/// The rerank decoder.
#[derive(Default)]
pub struct RerankDecoder {
    provider: &'static str,
}

impl Decoder<RerankOp> for RerankDecoder {
    type Event = RerankReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<RerankOp>) {
        let raw = serde_json::to_value(&event).unwrap_or(serde_json::Value::Null);
        let usage = event
            .usage
            .map(|usage| crate::completion::Usage {
                input_tokens: Some(usage.prompt_tokens),
                total_tokens: Some(usage.total_tokens),
                ..Default::default()
            })
            .unwrap_or_default();
        let results = event
            .results
            .into_iter()
            .map(|result| crate::rerank::RerankResult {
                index: result.index,
                document: result.document,
                relevance_score: result.relevance_score,
            })
            .collect();
        // A server that omits `model` still produced a ranking; `None` is the
        // honest report.
        out.push(Ok(crate::rerank::RerankResponse::new(results, self.provider)
            .with_optional_model(event.model)
            .with_usage(usage)
            .with_raw(raw)));
    }
}

impl Wire for Rerank {
    type Op = RerankOp;
    type Decoder = RerankDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn capabilities(&self) -> usize {
        self.provider.dialect.quirks.rerank.max_documents
    }

    fn encode(
        &self,
        request: crate::operation::RerankRequest,
        _mode: Mode,
    ) -> Result<Encoded, RerankError> {
        let quirks = &self.provider.dialect.quirks.rerank;
        // Empty means the dialect does not offer reranking, stated rather
        // than defaulted: a future dialect that leaves it out is refused
        // here instead of posting to a path its server never served.
        if quirks.path.is_empty() {
            return Err(RerankError::ProviderError(format!(
                "{} offers no reranking endpoint",
                self.provider.dialect.name
            )));
        }
        let mut body = serde_json::json!({
            "query": request.query,
            "documents": request.documents,
        });
        let Some(object) = body.as_object_mut() else {
            return Err(RerankError::ResponseError(
                "rerank request body must be an object".into(),
            ));
        };
        if quirks.sends_model_field {
            object.insert("model".to_owned(), serde_json::json!(self.model));
        }
        if let Some(top_n) = self.top_n {
            object.insert("top_n".to_owned(), serde_json::json!(top_n));
        }

        let request = json_request::<RerankError>(
            &self.provider,
            quirks.path,
            deployment(&self.provider, &self.model),
            &body,
        )?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> RerankDecoder {
        RerankDecoder {
            provider: self.provider.dialect.name,
        }
    }
}

// ── verification ────────────────────────────────────────────────────────

/// The credential-check wire: a `GET` whose status is the answer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Verify {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
}

impl Verify {
    /// The credential-check wire.
    pub fn new(provider: OpenAI) -> Self {
        Self { provider }
    }
}

/// The verification decoder.
///
/// The reply body is not read for meaning: a success status is the answer,
/// and the 401/403 classification lives in [`VerifyError`]'s `WireError`
/// impl, so this wire does not restate it.
#[derive(Default)]
pub struct VerifyDecoder;

impl Decoder<VerifyOp> for VerifyDecoder {
    type Event = ();

    fn classify(&self, _frame: WireFrame) -> WireEvent<Self::Event> {
        WireEvent::Known(())
    }

    fn interpret(&mut self, _event: Self::Event, out: &mut Output<VerifyOp>) {
        out.push(Ok(()));
    }

    fn finish(&mut self, out: &mut Output<VerifyOp>) {
        // A 200 with an empty body still verifies the credential: the status
        // is the whole answer, so the fold must not see an empty reply as a
        // missing payload.
        if out.items().is_empty() {
            out.push(Ok(()));
        }
    }
}

impl Wire for Verify {
    type Op = VerifyOp;
    type Decoder = VerifyDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, VerifyError> {
        let path = self.provider.dialect.quirks.verify_path;
        if path.is_empty() {
            return Err(VerifyError::ProviderError(format!(
                "{} offers no endpoint that checks a credential without consuming tokens",
                self.provider.dialect.name
            )));
        }
        let builder = http::Request::get(self.provider.uri(path, None));
        let request = self
            .provider
            .authenticate(builder)
            .body(Body::empty())
            .map_err(|error| VerifyError::ProviderError(error.to_string()))?;
        Ok(Encoded::new(request, Framing::Whole)
            .with_request_id_header(self.provider.dialect.request_id_header))
    }

    fn decoder(&self) -> VerifyDecoder {
        VerifyDecoder
    }
}

#[cfg(test)]
mod tests;
