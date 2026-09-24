//! Embedding, transcription, image, speech, listing, reranking, and verification wires.
//! Each operation consumes a whole reply, using its dialect's JSON or binary format.
//!
//! ```
//! use rig_core::providers::openai::{OpenAI, TEXT_EMBEDDING_3_SMALL};
//! let wire = OpenAI::new("key").embeddings(TEXT_EMBEDDING_3_SMALL, None);
//! ```

use serde::{Deserialize, Serialize};

use crate::embeddings;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::model::{Model, ModelList};
use crate::operation::{
    Embedding, EmbeddingCapabilities, ModelListing, Rerank as RerankOp, Transcription,
    Verify as VerifyOp,
};
use crate::providers::internal::wire::classify_untyped_line;
use crate::providers::openai::completion::Usage;
use crate::providers::openai::embedding::{
    CompatibleEmbeddingResponse, EncodingFormat, model_dimensions_from_identifier,
};
use crate::transcription::TranscriptionRequest;
use crate::wire::{
    Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame,
};

#[cfg(feature = "image")]
use super::ImageBody;
#[cfg(feature = "audio")]
use super::SpeechBody;
use super::{AcceptedWidths, ModelWidth, OpenAI, TranscriptionBody};

/// Encode an authenticated JSON POST with whole-response framing.
fn json_post(
    provider: &OpenAI,
    path: &str,
    deployment: Option<&str>,
    body: &serde_json::Value,
) -> Result<Encoded, EncodeError> {
    json_post_to(provider, provider.uri(path, deployment), body)
}

/// [`json_post`] against an already-resolved URL, for the endpoints whose
/// URL is derived rather than a fixed path under the base.
fn json_post_to(
    provider: &OpenAI,
    uri: String,
    body: &serde_json::Value,
) -> Result<Encoded, EncodeError> {
    let bytes = serde_json::to_vec(body)?;
    let builder = http::Request::post(uri).header("Content-Type", "application/json");
    encoded(provider, builder, Body::Bytes(bytes))
}

/// The `GET` whose status is the answer, for the two endpoints that send no
/// body: the model catalogue and the credential check.
fn get(provider: &OpenAI, path: &str) -> Result<Encoded, EncodeError> {
    encoded(
        provider,
        http::Request::get(provider.uri(path, None)),
        Body::empty(),
    )
}

/// Authenticate and build a request, then apply its modality envelope hook.
/// Return construction or hook errors. Use whole-response framing and the
/// dialect's request-ID header.
fn encoded(
    provider: &OpenAI,
    builder: http::request::Builder,
    body: Body,
) -> Result<Encoded, EncodeError> {
    let mut request = provider.authenticate(builder).body(body)?;
    if let Some(envelope) = provider
        .dialect
        .quirks
        .hooks
        .and_then(|hooks| hooks.modality_envelope)
    {
        envelope(provider, &mut request)?;
    }
    Ok(Encoded::new(request, Framing::Whole)
        .with_request_id_header(provider.dialect.request_id_header))
}

/// Refuse an embeddings request parameter the dialect does not accept.
fn unsupported_parameter(provider: &str, parameter: &str) -> EncodeError {
    EncodeError::request(format!(
        "{provider} embeddings do not support the `{parameter}` parameter"
    ))
}

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

    /// This model's width contract on this dialect, or `None` for a model
    /// the dialect does not document.
    fn model_width(&self) -> Option<&'static ModelWidth> {
        self.provider
            .dialect
            .quirks
            .embedding
            .widths
            .iter()
            .find(|width| width.model == self.model)
    }

    /// Resolve width from the caller, dialect table, then shared model table.
    /// Return zero when all are absent.
    fn resolved_ndims(&self) -> usize {
        self.ndims
            .or_else(|| self.model_width().and_then(|width| width.default))
            .or_else(|| model_dimensions_from_identifier(&self.model))
            .unwrap_or_default()
    }

    /// Validate declared widths against the dialect's zero-width and model policies.
    /// Return a request error for unsupported widths; unknown models are unchecked.
    fn refuse_unhonourable_width(&self) -> Result<(), EncodeError> {
        let quirks = &self.provider.dialect.quirks.embedding;
        let provider = self.provider.dialect.name;
        let invalid = |requirement, parameter| {
            EncodeError::request(format!(
                "{provider} embeddings require `{parameter}` {requirement}"
            ))
        };
        // A dialect that reads no width field has nothing to refuse: the
        // caller's number never reaches the wire, and the shared driver
        // catches the disagreement against the reply instead.
        let Some(parameter) = quirks.dimensions.name() else {
            return Ok(());
        };
        let Some(declared) = self.ndims else {
            return Ok(());
        };
        if declared == 0 {
            return match quirks.refuse_zero_width {
                Some(requirement) => Err(invalid(requirement, parameter)),
                None => Ok(()),
            };
        }
        // A model the dialect does not document: the caller's width is the
        // only width there is, so it goes out unvalidated and the API rules.
        let Some(width) = self.model_width() else {
            return Ok(());
        };
        // Native widths require no truncation parameter.
        if width.default == Some(declared) {
            return Ok(());
        }
        match width.accepted {
            AcceptedWidths::Fixed => Err(unsupported_parameter(provider, parameter)),
            AcceptedWidths::Range { min, max, .. } if (min..=max).contains(&declared) => Ok(()),
            AcceptedWidths::Range { requirement, .. } => Err(invalid(requirement, parameter)),
        }
    }

    /// The width to send, in the field this dialect spells it with.
    ///
    /// OpenAI's legacy Ada model does not accept a width at all, and
    /// `llama-server` reads no width field, so neither is sent one.
    fn requested_width(&self) -> Option<(&'static str, usize)> {
        let field = self.provider.dialect.quirks.embedding.dimensions.name()?;
        if self.model == crate::providers::openai::embedding::TEXT_EMBEDDING_ADA_002 {
            return None;
        }
        // Unknown widths are metadata sentinels, not request parameters.
        let ndims = match self.resolved_ndims() {
            0 => return None,
            ndims => ndims,
        };
        // At a documented model's native width, send nothing: that width is
        // what the model emits unasked, so the field would only restate the
        // default and the vector is identical either way.
        if self
            .model_width()
            .is_some_and(|width| width.default == Some(ndims))
        {
            return None;
        }
        Some((field, ndims))
    }
}

/// Decode embedding vectors and usage, enforcing the dialect's usage requirement.
#[derive(Default)]
pub struct EmbeddingsDecoder {
    /// Whether this dialect's reply must carry usage.
    requires_usage: bool,
    provider: &'static str,
    model: String,
}

impl Decoder<Embedding> for EmbeddingsDecoder {
    type Event = CompatibleEmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<Embedding>) {
        if event.usage.is_none() && self.requires_usage {
            out.push(Err(ProviderError::Response(format!(
                "{} embedding response omitted required usage",
                self.provider
            ))));
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
        // Only an explicit nonzero declaration permits reply-width mismatch checks.
        EmbeddingCapabilities::new(
            self.provider.dialect.quirks.embedding.max_documents,
            self.resolved_ndims(),
        )
        .declaring(self.ndims)
    }

    fn encode(&self, request: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let quirks = &self.provider.dialect.quirks.embedding;
        // Base64 vectors are not decoded anywhere, so asking for them would
        // answer 200 with a payload rig cannot read.
        if self.encoding_format == Some(EncodingFormat::Base64) {
            return Err(EncodeError::request(format!(
                "Rig cannot decode {} embedding responses encoded as `base64`",
                self.provider.dialect.name
            )));
        }
        if self.encoding_format.is_some() && !quirks.supports_encoding_format {
            return Err(unsupported_parameter(
                self.provider.dialect.name,
                "encoding_format",
            ));
        }
        if self.user.is_some() && !quirks.supports_user {
            return Err(unsupported_parameter(self.provider.dialect.name, "user"));
        }
        self.refuse_unhonourable_width()?;

        let mut body = serde_json::json!({ "input": request });
        let Some(object) = body.as_object_mut() else {
            return Err(EncodeError::request(
                "embedding request body must be an object",
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

        json_post(
            &self.provider,
            self.provider.dialect.quirks.embeddings_path,
            self.provider.deployment(&self.model),
            &body,
        )
    }

    fn decoder(&self, _mode: Mode) -> EmbeddingsDecoder {
        EmbeddingsDecoder {
            requires_usage: self.provider.dialect.quirks.embedding.requires_usage,
            provider: self.provider.dialect.name,
            model: self.model.clone(),
        }
    }
}

/// Transcription requests encoded as multipart or input-audio JSON.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transcriptions {
    /// Which provider, and how to reach it.
    pub provider: OpenAI,
    /// Transcription model or Azure deployment identifier.
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

    /// OpenAI's multipart upload: the audio as a file part beside the
    /// per-request options.
    fn multipart_body(&self, request: TranscriptionRequest) -> Result<Body, EncodeError> {
        use crate::http_client::MultipartForm;
        use crate::http_client::multipart::Part;

        let mut form = MultipartForm::new();
        // Azure addresses its deployment through the URL rather than the form.
        if self.provider.deployment(&self.model).is_none() {
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
            for (name, value) in additional_params_object(&additional_params)? {
                // Form strings must not acquire JSON quotation marks.
                let value = match value {
                    serde_json::Value::String(value) => value.clone(),
                    other => other.to_string(),
                };
                form = form.text(name.clone(), value);
            }
        }
        Ok(Body::Multipart(form))
    }

    /// Encode base64 audio under `input_audio`, inferring format from the filename.
    /// Reject top-level prompts and non-object additional parameters.
    fn input_audio_body(&self, request: TranscriptionRequest) -> Result<Body, EncodeError> {
        use base64::Engine;

        if request.prompt.is_some() {
            return Err(EncodeError::request(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "OpenRouter STT does not support a top-level prompt field. \
                     Provider-specific prompt options can be passed via `additional_params`. \
                     Example: {\"provider\": {\"options\": {\"<provider>\": {\"prompt\": \"<text>\"}}}}",
            )));
        }

        let mut body = serde_json::Map::new();
        body.insert("model".to_owned(), serde_json::json!(self.model));
        body.insert(
            "input_audio".to_owned(),
            serde_json::json!({
                "data": base64::engine::general_purpose::STANDARD.encode(&request.data),
                "format": audio_format_of(&request.filename),
            }),
        );
        if let Some(language) = request.language {
            body.insert("language".to_owned(), serde_json::json!(language));
        }
        if let Some(temperature) = request.temperature {
            body.insert("temperature".to_owned(), serde_json::json!(temperature));
        }
        if let Some(additional_params) = request.additional_params {
            for (name, value) in additional_params_object(&additional_params)? {
                body.insert(name.clone(), value.clone());
            }
        }
        Ok(Body::Bytes(serde_json::to_vec(
            &serde_json::Value::Object(body),
        )?))
    }
}

/// A transcription request's `additional_params`, as an object.
fn additional_params_object(
    params: &serde_json::Value,
) -> Result<&serde_json::Map<String, serde_json::Value>, EncodeError> {
    params.as_object().ok_or_else(|| {
        EncodeError::request(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "additional transcription parameters must be a JSON object",
        ))
    })
}

/// Infer the audio container from a case-insensitive filename extension.
/// Unknown or absent extensions default to `wav`.
fn audio_format_of(filename: &str) -> &'static str {
    let extension = std::path::Path::new(filename)
        .extension()
        .and_then(std::ffi::OsStr::to_str)
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("mp3") => "mp3",
        Some("flac") => "flac",
        Some("m4a") => "m4a",
        Some("ogg") => "ogg",
        Some("webm") => "webm",
        Some("aac") => "aac",
        _ => "wav",
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
            Err(error) => out.push(Err(ProviderError::from(error))),
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

    fn encode(&self, request: TranscriptionRequest, _mode: Mode) -> Result<Encoded, EncodeError> {
        let uri = self
            .provider
            .modality_uri(
                "transcription",
                self.provider.dialect.quirks.transcription_path,
                &self.model,
            )
            .map_err(EncodeError::request)?;
        let builder = http::Request::post(uri);
        let (builder, body) = match self.provider.dialect.quirks.transcription_body {
            TranscriptionBody::Multipart => (builder, self.multipart_body(request)?),
            TranscriptionBody::InputAudioJson => (
                builder.header(http::header::CONTENT_TYPE, "application/json"),
                self.input_audio_body(request)?,
            ),
        };
        encoded(&self.provider, builder, body)
    }

    fn decoder(&self, _mode: Mode) -> TranscriptionsDecoder {
        TranscriptionsDecoder {
            provider: self.provider.dialect.name,
        }
    }
}

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
///
/// Carries the dialect's [`ImageBody`] because the request shape names the
/// reply shape: most of this family answers with a JSON envelope, and
/// Hugging Face's router answers with the image bytes themselves.
#[cfg(feature = "image")]
#[derive(Default)]
pub struct ImagesDecoder {
    provider: &'static str,
    /// Which reply shape this dialect answers with.
    body: ImageBody,
}

/// One generated image, as every dialect on this wire returns it.
#[cfg(feature = "image")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDatum {
    /// The image, base64-encoded.
    pub b64_json: String,
}

/// Base64 image represented as a string or an object with an `image` field.
#[cfg(feature = "image")]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ImagesReplyImage {
    /// Hyperbolic: `{"image": "<base64>"}`.
    Keyed {
        /// The image, base64-encoded.
        image: String,
    },
    /// Venice: the base64 payload itself.
    Bare(String),
}

#[cfg(feature = "image")]
impl ImagesReplyImage {
    /// The image's base64 payload, whichever form the dialect sent.
    pub fn base64(&self) -> &str {
        match self {
            Self::Keyed { image } => image,
            Self::Bare(image) => image,
        }
    }
}

/// JSON image reply accepting `data` and `images` arrays with optional metadata.
#[cfg(feature = "image")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImagesReply {
    /// The generated images, as OpenAI and xAI key them.
    #[serde(default)]
    pub data: Vec<ImageDatum>,
    /// The generated images, as Hyperbolic and Venice key them.
    #[serde(default)]
    pub images: Vec<ImagesReplyImage>,
    /// Whatever else the dialect sent (`created`, Venice's `id`/`timing`, and
    /// any field this build does not model), so the raw payload loses
    /// nothing.
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

#[cfg(feature = "image")]
impl ImagesReply {
    /// The first image's base64 payload, whichever key the dialect used.
    pub fn first_base64(&self) -> Option<&str> {
        self.data
            .first()
            .map(|image| image.b64_json.as_str())
            .or_else(|| self.images.first().map(ImagesReplyImage::base64))
            .filter(|encoded| !encoded.is_empty())
    }
}

/// Generated images in a decoded JSON envelope or as raw bytes.
#[cfg(feature = "image")]
#[derive(Debug, Clone)]
pub enum ImagesEvent {
    /// A JSON envelope, as OpenAI, xAI and Hyperbolic answer.
    Json(ImagesReply),
    /// The image bytes themselves, with no envelope at all.
    Raw(Vec<u8>),
}

#[cfg(feature = "image")]
impl Decoder<crate::operation::ImageGeneration> for ImagesDecoder {
    type Event = ImagesEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        match self.body {
            // Preserve raw image bytes even when framing recognized valid UTF-8.
            ImageBody::HuggingFace => WireEvent::Known(ImagesEvent::Raw(match frame {
                WireFrame::Text(text) => text.into_bytes(),
                WireFrame::Bytes(bytes) => bytes,
            })),
            ImageBody::OpenAi | ImageBody::Xai | ImageBody::Hyperbolic | ImageBody::Venice => {
                classify_untyped_line(frame.as_str().as_bytes()).map(ImagesEvent::Json)
            }
        }
    }

    fn interpret(
        &mut self,
        event: Self::Event,
        out: &mut Output<crate::operation::ImageGeneration>,
    ) {
        use crate::image_generation::ImageGenerationResponse;
        use base64::Engine;

        let reply = match event {
            // The image is already the payload, and the reply is not a
            // document, so `raw` stays null rather than restating the bytes.
            ImagesEvent::Raw(image) => {
                out.push(Ok(ImageGenerationResponse::new(image, self.provider)));
                return;
            }
            ImagesEvent::Json(reply) => reply,
        };
        let Some(encoded) = reply.first_base64() else {
            out.push(Err(ProviderError::Response(
                "missing image data".to_owned(),
            )));
            return;
        };
        let image = match base64::prelude::BASE64_STANDARD.decode(encoded) {
            Ok(image) => image,
            Err(error) => {
                out.push(Err(ProviderError::Response(error.to_string())));
                return;
            }
        };
        let raw = serde_json::to_value(&reply).unwrap_or(serde_json::Value::Null);
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
    ) -> Result<Encoded, EncodeError> {
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
            ImageBody::Hyperbolic => serde_json::json!({
                "model_name": self.model,
                "prompt": request.prompt,
                "height": request.height,
                "width": request.width,
            }),
            ImageBody::Venice => serde_json::json!({
                "model": self.model,
                "prompt": request.prompt,
                "width": request.width,
                "height": request.height,
            }),
            // The model is addressed through the URL, not the body.
            ImageBody::HuggingFace => serde_json::json!({
                "inputs": request.prompt,
                "parameters": {
                    "width": request.width,
                    "height": request.height,
                },
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
            .map_err(EncodeError::request)?;
        json_post_to(&self.provider, uri, &body)
    }

    fn decoder(&self, _mode: Mode) -> ImagesDecoder {
        ImagesDecoder {
            provider: self.provider.dialect.name,
            body: self.provider.dialect.quirks.image_body,
        }
    }
}

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

/// Decode raw audio or the dialect's base64 JSON envelope.
#[cfg(feature = "audio")]
#[derive(Default)]
pub struct SpeechDecoder {
    provider: &'static str,
    /// Which reply shape this dialect answers with.
    body: SpeechBody,
}

/// Hyperbolic's speech reply: base64 in a JSON envelope rather than the
/// audio bytes themselves.
#[cfg(feature = "audio")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechReply {
    /// The audio, base64-encoded.
    pub audio: String,
}

#[cfg(feature = "audio")]
impl Decoder<crate::operation::AudioGeneration> for SpeechDecoder {
    type Event = Vec<u8>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        // Interpretation selects raw audio or JSON decoding from the dialect.
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
        use base64::Engine;

        let audio = match self.body {
            // OpenAI and xAI answer with the audio itself.
            SpeechBody::OpenAi | SpeechBody::Xai => event,
            // Hyperbolic wraps it, base64-encoded, in a JSON envelope.
            SpeechBody::Hyperbolic => {
                let reply = match serde_json::from_slice::<SpeechReply>(&event) {
                    Ok(reply) => reply,
                    Err(error) => {
                        out.push(Err(ProviderError::Response(error.to_string())));
                        return;
                    }
                };
                match base64::prelude::BASE64_STANDARD.decode(&reply.audio) {
                    Ok(audio) => audio,
                    Err(error) => {
                        out.push(Err(ProviderError::Response(error.to_string())));
                        return;
                    }
                }
            }
        };
        out.push(Ok(crate::audio_generation::AudioGenerationResponse::new(
            audio,
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
    ) -> Result<Encoded, EncodeError> {
        let mut body = match self.provider.dialect.quirks.speech_body {
            SpeechBody::OpenAi => serde_json::json!({
                "model": self.model,
                "input": request.text,
                "voice": request.voice,
                "speed": request.speed,
            }),
            // xAI requires a default voice when the caller leaves it empty.
            SpeechBody::Xai => serde_json::json!({
                "text": request.text,
                "voice_id": if request.voice.is_empty() { "eve" } else { request.voice.as_str() },
                "language": "en",
            }),
            // Hyperbolic addresses this endpoint by language, so the
            // identifier the caller passes as the model IS the language tag.
            SpeechBody::Hyperbolic => serde_json::json!({
                "language": self.model,
                "speaker": request.voice,
                "text": request.text,
                "speed": request.speed,
            }),
        };
        // Caller parameters take precedence, including response format and instructions.
        if let Some(additional_params) = request.additional_params {
            crate::json_utils::merge_inplace(&mut body, additional_params);
        }

        // Azure versions its speech endpoint separately from every other
        // route, so this one request carries its own `api-version`.
        let uri = self.provider.uri_versioned(
            self.provider.dialect.quirks.audio_generation_path,
            self.provider.deployment(&self.model),
            self.provider.speech_api_version(),
        );
        json_post_to(&self.provider, uri, &body)
    }

    fn decoder(&self, _mode: Mode) -> SpeechDecoder {
        SpeechDecoder {
            provider: self.provider.dialect.name,
            body: self.provider.dialect.quirks.speech_body,
        }
    }
}

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

/// Model-list entry with required `id` and optional metadata.
/// Context length prefers `context_window`, then `context_length`, then
/// `max_context_length`. Top-level output limits take precedence over `top_provider`.
#[derive(Debug, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    /// Mistral labels the model kind `type` (`base`, `fine-tuned`).
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    #[serde(default)]
    pub created: Option<u64>,
    #[serde(default)]
    pub owned_by: Option<String>,
    #[serde(default)]
    pub context_window: Option<u32>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub max_context_length: Option<u32>,
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
    #[serde(default)]
    pub top_provider: Option<TopProvider>,
}

/// OpenRouter's per-entry routing block. Only the output ceiling is read;
/// the rest of the block is routing detail [`Model`] has no slot for.
#[derive(Debug, Deserialize)]
pub struct TopProvider {
    #[serde(default)]
    pub max_completion_tokens: Option<u32>,
}

impl From<ModelEntry> for Model {
    fn from(entry: ModelEntry) -> Self {
        let mut model = Model::from_id(entry.id);
        model.name = entry.name;
        model.description = entry.description;
        model.r#type = entry.kind;
        model.created_at = entry.created;
        model.owned_by = entry.owned_by;
        model.context_length = entry
            .context_window
            .or(entry.context_length)
            .or(entry.max_context_length);
        model.max_output_tokens = entry.max_completion_tokens.or_else(|| {
            entry
                .top_provider
                .and_then(|provider| provider.max_completion_tokens)
        });
        model
    }
}

/// The `{ "data": [...] }` envelope.
#[derive(Debug, Deserialize)]
pub struct ModelsReply {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

/// Decode a complete model catalogue without pagination.
#[derive(Default)]
pub struct ModelsDecoder;

impl Decoder<ModelListing> for ModelsDecoder {
    type Event = ModelsReply;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_untyped_line(frame.as_str().as_bytes())
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<ModelListing>) {
        let models = event.data.into_iter().map(Model::from).collect();
        out.push(Ok(ModelList::new(models)));
    }
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        get(&self.provider, self.provider.dialect.quirks.models_path)
    }

    fn decoder(&self, _mode: Mode) -> ModelsDecoder {
        ModelsDecoder
    }
}

/// Rerank documents with `{model, query, documents, top_n}` requests.
/// The dialect must configure a nonempty reranking path.
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

/// Scored input document accepting `relevance_score` or `score` as its score key.
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
        out.push(Ok(crate::rerank::RerankResponse::new(
            results,
            self.provider,
        )
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
    ) -> Result<Encoded, EncodeError> {
        let quirks = &self.provider.dialect.quirks.rerank;
        // An empty path explicitly disables reranking.
        if quirks.path.is_empty() {
            return Err(EncodeError::request(format!(
                "{} offers no reranking endpoint",
                self.provider.dialect.name
            )));
        }
        let mut body = serde_json::json!({
            "query": request.query,
            "documents": request.documents,
        });
        let Some(object) = body.as_object_mut() else {
            return Err(EncodeError::request(
                "rerank request body must be an object",
            ));
        };
        if quirks.sends_model_field {
            object.insert("model".to_owned(), serde_json::json!(self.model));
        }
        if let Some(top_n) = self.top_n {
            object.insert("top_n".to_owned(), serde_json::json!(top_n));
        }

        json_post(
            &self.provider,
            quirks.path,
            self.provider.deployment(&self.model),
            &body,
        )
    }

    fn decoder(&self, _mode: Mode) -> RerankDecoder {
        RerankDecoder {
            provider: self.provider.dialect.name,
        }
    }
}

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

pub use crate::operation::VerifyDecoder;

impl Wire for Verify {
    type Op = VerifyOp;
    type Decoder = VerifyDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let path = self.provider.dialect.quirks.verify_path;
        if path.is_empty() {
            return Err(EncodeError::request(format!(
                "{} offers no endpoint that checks a credential without consuming tokens",
                self.provider.dialect.name
            )));
        }
        get(&self.provider, path)
    }

    fn decoder(&self, _mode: Mode) -> VerifyDecoder {
        VerifyDecoder
    }
}

#[cfg(test)]
mod tests;
