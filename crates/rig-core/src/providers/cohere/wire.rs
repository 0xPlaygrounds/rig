//! Cohere configuration and chat, text-embedding, and image-embedding wires.
//!
//! ```no_run
//! use rig_core::providers::cohere::{Cohere, EMBED_V4};
//! let wire = Cohere::from_env()?.embedding(EMBED_V4);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::client::env::{self, EnvError};
use crate::completion::CompletionRequest;
use crate::embeddings::Embedding as Vector;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::json_utils;
use crate::operation::{Completion, Embedding, ImageEmbedding};
use crate::wire::Flow;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Secret, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::completion::{CohereCompletionRequest, PROVIDER_NAME};
use crate::message::Issuer;

/// The issuer of Cohere's reasoning, which is the only reasoning it replays.
pub(crate) const ISSUER: Issuer = Issuer::from_static(PROVIDER_NAME);
use super::embeddings::{
    EmbeddingResponse as CohereEmbeddingResponse, ErrorEnvelope as CohereErrorEnvelope,
    ImageEmbeddingResponse as CohereImageEmbeddingResponse, image_data_url, validate_image,
};
use super::streaming::ChatDecoder;

/// Cohere's API root.
const BASE_URL: &str = "https://api.cohere.ai";

/// The environment variable carrying the API key.
const API_KEY_ENV: &str = "COHERE_API_KEY";

/// The settings of a Cohere provider: serializable, and the credential is
/// never serialized. [`connect`](Self::connect) puts it on a transport as a
/// [`Cohere`](super::Cohere) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohereConfig {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl CohereConfig {
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
    pub(crate) fn completion(&self, model: impl Into<String>) -> Chat {
        Chat {
            provider: self.clone(),
            model: model.into(),
        }
    }

    /// The text-embedding wire for `model`, at the model's native width.
    pub(crate) fn embedding(&self, model: impl Into<String>) -> Embeddings {
        Embeddings {
            provider: self.clone(),
            model: model.into(),
            ndims: None,
            input_type: DEFAULT_INPUT_TYPE.to_owned(),
        }
    }

    /// The image-embedding wire.
    ///
    /// Cohere Embed v3 embeds images with one fixed model, so this wire
    /// names no model.
    pub(crate) fn image_embedding(&self) -> ImageEmbeddings {
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
    pub provider: CohereConfig,
    /// The model to address.
    pub model: String,
}

impl Wire for Chat {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ChatDecoder<'id>;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME).model(self.model.as_str())
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let request = request.replayable_to(&[ISSUER])?;
        let mut body = CohereCompletionRequest::try_from((self.model.as_str(), request))?;
        if mode == Mode::Streaming {
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
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
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

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ChatDecoder::default()
    }
}

/// Default retrieval role for embeddings of stored document chunks.
const DEFAULT_INPUT_TYPE: &str = "search_document";

/// The most texts Cohere embeds in one `/v1/embed` call.
const MAX_DOCUMENTS: usize = 96;

/// The width Cohere's image embeddings come back at.
const IMAGE_NDIMS: usize = 1_024;

/// Recognized embedding and error-envelope markers, including errors on HTTP 200.
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

/// Decode embeddings, then an error envelope on failure. Retain the embedding
/// diagnostic if neither shape decodes.
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
    pub provider: CohereConfig,
    /// The model to address.
    pub model: String,
    /// The width the caller declared through
    /// [`EmbeddingWidth`](crate::embeddings::EmbeddingWidth), if any.
    pub ndims: Option<usize>,
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

/// The embed endpoint takes no width, so nothing is sent: declare the width
/// of a model the table does not know, and the reply is checked against it.
impl crate::embeddings::EmbeddingWidth for Embeddings {
    fn with_ndims(mut self, ndims: usize) -> Self {
        self.ndims = Some(ndims);
        self
    }
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = EmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(
                Capabilities::embedding(
                    MAX_DOCUMENTS,
                    self.ndims
                        .or_else(|| super::model_dimensions_from_identifier(&self.model))
                        .unwrap_or_default(),
                )
                .declaring(self.ndims),
            )
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::json!({
            "model": self.model,
            "texts": texts,
            "input_type": self.input_type,
        });
        let request = self
            .provider
            .post("/v1/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder
    }
}

/// Decodes one `/v1/embed` reply for texts.
pub struct EmbeddingsDecoder;

impl<'id> Decoder<'id, Embedding> for EmbeddingsDecoder {
    type Event = EmbedReply<CohereEmbeddingResponse>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_embed_reply(&frame.as_str())
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, Embedding>,
    ) -> Result<Flow, ProviderError> {
        let reply = match reply {
            EmbedReply::Reply(reply) => reply,
            // Preserve the error body so the driver can attach its HTTP status.
            EmbedReply::Failure(body) => {
                return Err(ProviderError::from_provider_body(body));
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
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            response_id: Some(reply.id),
            usage,
            ..crate::embeddings::EmbeddingResponse::new(vectors)
        }))
    }
}

/// The image-embedding wire: `POST /v1/embed`, one image per request.
///
/// Cohere Embed v3 accepts a single image per call, so the wire's batch limit
/// is one image.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageEmbeddings {
    /// The provider this wire speaks to.
    pub provider: CohereConfig,
}

impl Wire for ImageEmbeddings {
    type Op = ImageEmbedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ImageEmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(super::EMBED_ENGLISH_V3)
            .capabilities(Capabilities::embedding(1, IMAGE_NDIMS))
    }

    fn encode(&self, images: Vec<Vec<u8>>, _mode: Mode) -> Result<Encoded, EncodeError> {
        // The wire's batch limit is one image: a caller splits larger
        // batches into one call per image.
        let [image] = images.as_slice() else {
            return Err(EncodeError::request(format!(
                "Cohere embeds one image per request, not {}",
                images.len()
            )));
        };
        let media_type = validate_image(image)?;
        let body = serde_json::json!({
            "model": super::EMBED_ENGLISH_V3,
            "images": [image_data_url(image, media_type)],
            "input_type": "image",
            "embedding_types": ["float"],
        });
        let request = self
            .provider
            .post("/v1/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ImageEmbeddingsDecoder
    }
}

/// Decodes one `/v1/embed` reply for a single image.
pub struct ImageEmbeddingsDecoder;

impl<'id> Decoder<'id, ImageEmbedding> for ImageEmbeddingsDecoder {
    type Event = EmbedReply<CohereImageEmbeddingResponse>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_embed_reply(&frame.as_str())
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, ImageEmbedding>,
    ) -> Result<Flow, ProviderError> {
        let reply = match reply {
            EmbedReply::Reply(reply) => reply,
            // Same 200-with-an-envelope reply as the text route: the body
            // verbatim, with the driver stamping the status.
            EmbedReply::Failure(body) => {
                return Err(ProviderError::from_provider_body(body));
            }
        };
        // Each request carries one image, so any other vector count is invalid.
        let [vector] = reply.embeddings.values.as_slice() else {
            return Err(ProviderError::Response(format!(
                "Expected 1 image embedding, got {}",
                reply.embeddings.values.len()
            )));
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
        Ok(out.end(crate::embeddings::ImageEmbeddingResponse {
            usage,
            response_id: reply.id,
            ..crate::embeddings::ImageEmbeddingResponse::new(vec![vector])
        }))
    }
}

#[cfg(test)]
mod tests;
