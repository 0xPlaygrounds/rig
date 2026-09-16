//! The unary modality operations: embeddings, reranking, transcription,
//! image generation, audio generation.
//!
//! Each is the same shape — one request, one reply document, one normalized
//! response carrying usage and identity — so they are declared once by
//! [`modality_operation!`] rather than five times by hand. The only
//! per-operation data is the request, the response, the error enum and the
//! canonical telemetry name.

use super::{One, Take};
use crate::telemetry::{ModalityOperation, ModalityResponseTelemetry, SpanCombinator};
use crate::wire::{Operation, Reply};

/// What a runtime accounts for on an embedding wire: the batch limit the
/// provider accepts and the dimensionality it returns.
///
/// These are facts about the wire that the consumer trait asks for
/// (`max_documents`, `ndims`), so they ride on the operation's capability
/// value rather than on extra trait methods no other operation would want.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbeddingCapabilities {
    /// The most documents the provider embeds in one request.
    pub max_documents: usize,
    /// The dimensionality of the returned vectors.
    pub ndims: usize,
}

impl EmbeddingCapabilities {
    /// The capability pair for a wire.
    pub const fn new(max_documents: usize, ndims: usize) -> Self {
        Self {
            max_documents,
            ndims,
        }
    }
}

/// A reranking request: the query, the documents to order, and the batch
/// limit's subject.
///
/// The consumer trait takes `(&str, Vec<String>)`; the operation takes one
/// request value, as every other operation does.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankRequest {
    /// What the documents are ordered against.
    pub query: String,
    /// The documents to order, in input order.
    pub documents: Vec<String>,
}

/// Declare one unary modality operation.
macro_rules! modality_operation {
    (
        $(#[$doc:meta])*
        $op:ident {
            request: $request:ty,
            response: $response:ty,
            error: $error:ty,
            capabilities: $capabilities:ty,
            telemetry: $telemetry:ident,
            name: $name:literal,
        }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $op;

        impl Operation for $op {
            type Request = $request;
            type Event = $response;
            type Response = $response;
            type Error = $error;
            type Capabilities = $capabilities;
            type Output = One<Self>;
            type Fold = Take<Self>;
            type Telemetry = ModalityOperation;

            const NAME: &'static str = $name;

            fn is_terminal(_event: &Self::Event) -> bool {
                true
            }

            fn telemetry(_streaming: bool) -> Self::Telemetry {
                ModalityOperation::$telemetry
            }

            fn stamp_reply(response: &mut Self::Response, reply: Reply) {
                if response.provider_request_id.is_none() {
                    response.provider_request_id = reply.provider_request_id;
                }
                if response.raw.is_null() {
                    response.raw = reply.raw;
                }
            }

            fn span(
                provider: &str,
                model: Option<&str>,
                telemetry: Self::Telemetry,
                _request: &Self::Request,
            ) -> tracing::Span {
                crate::telemetry::ModalitySpanBuilder::new(
                    provider,
                    model.unwrap_or_default(),
                    telemetry,
                )
                .build()
            }

            fn record(span: &tracing::Span, response: &Self::Response) {
                if span.is_disabled() {
                    return;
                }
                span.record_token_usage(response.telemetry_usage());
                if let Some(id) = response.telemetry_response_id() {
                    span.record("gen_ai.response.id", id);
                }
                if let Some(model) = response.telemetry_model() {
                    span.record("gen_ai.response.model", model);
                }
            }
        }
    };
}

modality_operation!(
    /// Embedding a batch of texts.
    Embedding {
        request: Vec<String>,
        response: crate::embeddings::EmbeddingResponse,
        error: crate::embeddings::EmbeddingError,
        capabilities: EmbeddingCapabilities,
        telemetry: Embeddings,
        name: "embedding",
    }
);

modality_operation!(
    /// Embedding a batch of images from their encoded file bytes.
    ImageEmbedding {
        request: Vec<Vec<u8>>,
        response: crate::embeddings::ImageEmbeddingResponse,
        error: crate::embeddings::EmbeddingError,
        capabilities: EmbeddingCapabilities,
        telemetry: Embeddings,
        name: "image_embedding",
    }
);

modality_operation!(
    /// Ordering documents by relevance to a query.
    Rerank {
        request: RerankRequest,
        response: crate::rerank::RerankResponse,
        error: crate::rerank::RerankError,
        capabilities: usize,
        telemetry: Rerank,
        name: "rerank",
    }
);

modality_operation!(
    /// Transcribing audio. The only operation whose request is multipart.
    Transcription {
        request: crate::transcription::TranscriptionRequest,
        response: crate::transcription::TranscriptionResponse,
        error: crate::transcription::TranscriptionError,
        capabilities: (),
        telemetry: Transcription,
        name: "transcription",
    }
);

#[cfg(feature = "image")]
modality_operation!(
    /// Generating an image.
    ImageGeneration {
        request: crate::image_generation::ImageGenerationRequest,
        response: crate::image_generation::ImageGenerationResponse,
        error: crate::image_generation::ImageGenerationError,
        capabilities: (),
        telemetry: ImageGeneration,
        name: "image_generation",
    }
);

#[cfg(feature = "audio")]
modality_operation!(
    /// Generating speech.
    AudioGeneration {
        request: crate::audio_generation::AudioGenerationRequest,
        response: crate::audio_generation::AudioGenerationResponse,
        error: crate::audio_generation::AudioGenerationError,
        capabilities: (),
        telemetry: AudioGeneration,
        name: "audio_generation",
    }
);
