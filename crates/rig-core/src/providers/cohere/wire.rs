//! Cohere as data: one config struct, three wires.
//!
//! Wires: [`Chat`] (`Completion`), [`Embeddings`] (`Embedding`) and
//! [`ImageEmbeddings`] (`ImageEmbedding`) — the three rows Cohere serves in
//! this crate. Cohere has one dialect, so its defaults are the config's
//! defaults rather than a table of constants.

use crate::client::env::{self, EnvError};
use crate::completion::{CompletionError, CompletionRequest};
use crate::driver::{HasEmbedding, HasImageEmbedding};
use crate::embeddings::{Embedding as Vector, EmbeddingError};
use crate::json_utils;
use crate::operation::{Completion, Embedding, EmbeddingCapabilities, ImageEmbedding};
use crate::wire::{
    Body, Decoder, Encoded, Framing, HasCompletion, Mode, Output, Secret, Sink, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::completion::{CohereCompletionRequest, PROVIDER_NAME};
use super::embeddings::{
    EmbeddingResponse as CohereEmbeddingResponse, ErrorEnvelope as CohereErrorEnvelope,
    ImageEmbeddingResponse as CohereImageEmbeddingResponse, image_data_url, validate_image,
};
use super::streaming::ChatDecoder;

/// Cohere's API root.
const BASE_URL: &str = "https://api.cohere.ai";

/// The environment variable carrying the API key.
const API_KEY_ENV: &str = "COHERE_API_KEY";

/// The shared configuration of a Cohere provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cohere {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl Cohere {
    /// Cohere with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_owned(),
        }
    }

    /// Cohere from `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Point the wires at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// The chat wire for `model`.
    pub fn chat(&self, model: impl Into<String>) -> Chat {
        Chat {
            provider: self.clone(),
            model: model.into(),
        }
    }

    /// The text-embedding wire for `model`, at `ndims` dimensions when the
    /// caller named one rather than taking the model's published width.
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        let model = model.into();
        let ndims = ndims
            .or_else(|| super::model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Embeddings {
            provider: self.clone(),
            model,
            ndims,
            input_type: DEFAULT_INPUT_TYPE.to_owned(),
        }
    }

    /// The image-embedding wire.
    ///
    /// Cohere Embed v3 embeds images with one fixed model, so this wire
    /// names no model.
    pub fn image_embeddings(&self) -> ImageEmbeddings {
        ImageEmbeddings {
            provider: self.clone(),
        }
    }

    /// One request, authenticated and typed as JSON.
    fn post(&self, path: &str) -> http::request::Builder {
        http::Request::post(format!("{}{path}", self.base_url))
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose()),
            )
    }
}

/// The chat wire: `POST /v2/chat`, SSE when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    /// The provider this wire speaks to.
    pub provider: Cohere,
    /// The model to address.
    pub model: String,
}

impl Wire for Chat {
    type Op = Completion;
    type Decoder = ChatDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        let mut body = CohereCompletionRequest::try_from((self.model.as_str(), request))?;
        if mode == Mode::Streaming {
            // Cohere streams the same endpoint; `stream` is the only
            // difference between the two requests, and the recorded bodies
            // pin it.
            body.additional_params = Some(json_utils::merge(
                body.additional_params
                    .take()
                    .unwrap_or_else(|| serde_json::json!({})),
                serde_json::json!({ "stream": true }),
            ));
        }
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Cohere completion request",
            &body,
        );
        let request = self
            .provider
            .post("/v2/chat")
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|error| CompletionError::HttpError(error.into()))?;
        // Cohere reports no request-id response header: its
        // `x-debug-trace-id` is a debug trace handle, not a documented
        // request id, so the normalized id stays unset by design.
        Ok(Encoded::new(
            request,
            match mode {
                Mode::Unary => Framing::Whole,
                Mode::Streaming => Framing::Sse,
            },
        ))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ChatDecoder::default()
    }
}

/// What `input_type` an embedding wire declares unless the caller names
/// another: Cohere's asymmetric embed models prompt differently for stored
/// chunks and for queries, and the stored-chunk side is the default the
/// deleted client used.
const DEFAULT_INPUT_TYPE: &str = "search_document";

/// The most texts Cohere embeds in one `/v1/embed` call.
const MAX_DOCUMENTS: usize = 96;

/// The width Cohere's image embeddings come back at.
const IMAGE_NDIMS: usize = 1_024;

/// The top-level keys that recognize a `/v1/embed` reply: `embeddings`, the
/// field every answer carries, and `message` — a 200 whose body is nothing
/// but Cohere's error envelope is a reply too, and recognizing it here is
/// what makes the answer's typed decode fail and hands the frame to the
/// envelope classifier.
const EMBED_REPLY_MARKERS: &[&str] = &["embeddings", "message"];

/// The key that recognizes the error envelope on its own.
const EMBED_ERROR_MARKERS: &[&str] = &["message"];

/// One `/v1/embed` reply: the answer, or the error envelope Cohere can
/// answer a **200** with instead.
pub enum EmbedReply<T> {
    /// The vectors Cohere returned.
    Reply(T),
    /// The provider's error envelope, verbatim.
    Failure(String),
}

/// Classify one `/v1/embed` reply, on either embed route.
///
/// Two shapes on one decoder, composed through the classify layer's own
/// combinator ([`classify_or`]) so no triage verdict is read here: the
/// answer is the shape the wire mostly sends and stays the diagnostic when
/// neither decodes, and the envelope is tried exactly when the answer's
/// decode fails.
///
/// [`classify_or`]: crate::providers::internal::wire::classify_or
fn classify_embed_reply<T>(data: &str) -> WireEvent<EmbedReply<T>>
where
    T: serde::de::DeserializeOwned,
{
    crate::providers::internal::wire::classify_or(
        data,
        |data| {
            crate::providers::internal::wire::classify_marker_keyed_frame::<T>(
                data,
                EMBED_REPLY_MARKERS,
            )
            .map(EmbedReply::Reply)
        },
        |data| {
            crate::providers::internal::wire::classify_marker_keyed_frame::<CohereErrorEnvelope>(
                data,
                EMBED_ERROR_MARKERS,
            )
            .map(|_| EmbedReply::Failure(data.to_owned()))
        },
    )
}

/// The text-embedding wire: `POST /v1/embed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The provider this wire speaks to.
    pub provider: Cohere,
    /// The model to address.
    pub model: String,
    /// The width this wire reports, from the caller or the model's published
    /// dimensions. `0` means neither named one.
    pub ndims: usize,
    /// Cohere's retrieval prompt: `search_document` for stored chunks,
    /// `search_query` for queries, `classification`, `clustering`.
    pub input_type: String,
}

impl Embeddings {
    /// Embed for a different purpose than storing chunks.
    pub fn with_input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = input_type.into();
        self
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

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EmbeddingError> {
        let body = serde_json::json!({
            "model": self.model,
            "texts": texts,
            "input_type": self.input_type,
        });
        let request = self
            .provider
            .post("/v1/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))
            .map_err(|error| EmbeddingError::HttpError(error.into()))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        EmbeddingsDecoder
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(MAX_DOCUMENTS, self.ndims)
    }
}

/// Decodes one `/v1/embed` reply for texts.
pub struct EmbeddingsDecoder;

impl Decoder<Embedding> for EmbeddingsDecoder {
    type Event = EmbedReply<CohereEmbeddingResponse>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_embed_reply(&frame.as_str())
    }

    fn interpret(&mut self, reply: Self::Event, out: &mut Output<Embedding>) {
        let reply = match reply {
            EmbedReply::Reply(reply) => reply,
            // A 200 that carried the error envelope instead of vectors: the
            // provider's body verbatim, and the driver stamps the status it
            // arrived under. Without this the frame read as an unmodeled
            // event, the driver warn-skipped it, and the fold failed with
            // "Expected 1 embeddings, got 0" — no status, no body.
            EmbedReply::Failure(body) => {
                out.push(Err(EmbeddingError::from_provider_body(body)));
                return;
            }
        };
        let raw = match serde_json::to_value(&reply) {
            Ok(raw) => raw,
            Err(error) => {
                out.push(Err(error.into()));
                return;
            }
        };
        let usage = reply
            .meta
            .as_ref()
            .map(|meta| meta.billed_units.to_usage())
            .unwrap_or_default();
        // The vectors only; the operation's fold pairs them with the texts
        // that were sent, which no `/v1/embed` reply is trusted to echo
        // back in order.
        let vectors = reply
            .embeddings
            .into_iter()
            .map(|vector| Vector {
                document: String::new(),
                vec: vector.into_iter().filter_map(|n| n.as_f64()).collect(),
            })
            .collect();
        // Cohere's `/v1/embed` reply names no model.
        out.push(Ok(crate::embeddings::EmbeddingResponse::new(
            vectors,
            PROVIDER_NAME,
        )
        .with_response_id(reply.id)
        .with_usage(usage)
        .with_raw(raw)));
    }
}

/// The image-embedding wire: `POST /v1/embed`, one image per request.
///
/// Cohere Embed v3 accepts a single image per call, so a batch is a batch of
/// requests ([`Encoded::batch`]) whose replies the operation's fold
/// concatenates in input order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageEmbeddings {
    /// The provider this wire speaks to.
    pub provider: Cohere,
}

impl Wire for ImageEmbeddings {
    type Op = ImageEmbedding;
    type Decoder = ImageEmbeddingsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(super::EMBED_ENGLISH_V3)
    }

    fn encode(&self, images: Vec<Vec<u8>>, _mode: Mode) -> Result<Encoded, EmbeddingError> {
        let requests = images
            .into_iter()
            .map(|image| {
                // Cohere's acceptance policy runs before the request is
                // built: an unsniffable or oversized image is a caller
                // error, not a 400 to interpret.
                let media_type = validate_image(&image)?;
                let body = serde_json::json!({
                    "model": super::EMBED_ENGLISH_V3,
                    "images": [image_data_url(&image, media_type)],
                    "input_type": "image",
                    "embedding_types": ["float"],
                });
                self.provider
                    .post("/v1/embed")
                    .body(Body::Bytes(serde_json::to_vec(&body)?))
                    .map_err(|error| EmbeddingError::HttpError(error.into()))
            })
            .collect::<Result<Vec<_>, EmbeddingError>>()?;
        Ok(Encoded::batch(requests, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ImageEmbeddingsDecoder
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(1, IMAGE_NDIMS)
    }
}

/// Decodes one `/v1/embed` reply for a single image.
pub struct ImageEmbeddingsDecoder;

impl Decoder<ImageEmbedding> for ImageEmbeddingsDecoder {
    type Event = EmbedReply<CohereImageEmbeddingResponse>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_embed_reply(&frame.as_str())
    }

    fn interpret(&mut self, reply: Self::Event, out: &mut Output<ImageEmbedding>) {
        let reply = match reply {
            EmbedReply::Reply(reply) => reply,
            // Same 200-with-an-envelope reply as the text route: the body
            // verbatim, with the driver stamping the status.
            EmbedReply::Failure(body) => {
                out.push(Err(EmbeddingError::from_provider_body(body)));
                return;
            }
        };
        // One image per request, so one vector per reply: a second one is a
        // provider defect, and none means the request bought nothing.
        let [vector] = reply.embeddings.values.as_slice() else {
            out.push(Err(EmbeddingError::ResponseError(format!(
                "Expected 1 image embedding, got {}",
                reply.embeddings.values.len()
            ))));
            return;
        };
        let usage = reply
            .meta
            .as_ref()
            .map(|meta| meta.billed_units.to_usage())
            .unwrap_or_default();
        let vector = Vector {
            // The fold names the input: an image has no text, and its bytes
            // must never travel back in a response.
            document: String::new(),
            vec: vector.iter().filter_map(|n| n.as_f64()).collect(),
        };
        // No `with_raw` here: this wire answers one operation with several
        // requests, so a per-reply capture would be the first page's document
        // and the fold keeps the first metadata it is given — every later page
        // would be unreachable. The driver owns `raw` for a batch: it collects
        // each page's verbatim document and stamps the sequence (the bare
        // document when there was only one page).
        let mut response =
            crate::embeddings::ImageEmbeddingResponse::new(vec![vector], PROVIDER_NAME)
                .with_usage(usage);
        if let Some(id) = reply.id {
            response = response.with_response_id(id);
        }
        out.push(Ok(response));
    }
}

impl HasCompletion for Cohere {
    type Wire = Chat;

    fn completion(&self, model: impl Into<String>) -> Chat {
        self.chat(model)
    }
}

impl HasEmbedding for Cohere {
    type Wire = Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        self.embeddings(model, ndims)
    }
}

impl HasImageEmbedding for Cohere {
    type Wire = ImageEmbeddings;

    fn image_embedding(&self, _model: impl Into<String>, _ndims: Option<usize>) -> ImageEmbeddings {
        // Cohere Embed v3 embeds images with one fixed model at one fixed
        // width, so neither argument has anywhere to go.
        self.image_embeddings()
    }
}

#[cfg(test)]
mod tests;
