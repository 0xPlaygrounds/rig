//! Buffered embedding, reranking, transcription, image, and audio operations.
//!
//! ```
//! use rig_core::wire::Capabilities;
//!
//! let capabilities = Capabilities::embedding(32, 768).declaring(Some(768));
//! assert_eq!(capabilities.declared, Some(768));
//! ```

use std::convert::Infallible;

use super::Whole;
use crate::embeddings::Embedding as Vector;
use crate::error::ProviderError;
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Call, Capabilities, Fold, Free, Operation, Reply};

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
            telemetry: $telemetry:ident,
            fold: $fold:ty,
            seed: $seed:expr,
        }
    ) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $op;

        impl Operation for $op {
            type Request = $request;
            type Event = Infallible;
            type End = $response;
            type Response = $response;
            type Fold = Traced<Self, $fold>;
            type Emit = Free;

            fn fold(request: &Self::Request, call: &mut Call<'_>) -> Self::Fold {
                let telemetry = call
                    .wire
                    .telemetry
                    .map_or(GenAiOperation::$telemetry, |telemetry| telemetry(call.mode));
                debug_assert!(!telemetry.is_completion());
                let span = SpanBuilder::new(
                    call.wire.name,
                    call.wire.model.unwrap_or_default(),
                    telemetry,
                )
                .build();
                call.instrument(span.clone());
                #[allow(clippy::redundant_closure_call)]
                let inner = ($seed)(request, &*call);
                Traced {
                    inner,
                    span,
                    record: |span, response| {
                        span.record_response(
                            response.response_id.as_deref(),
                            response.model.as_deref(),
                            &response.usage,
                        )
                    },
                }
            }
        }
    };
}

/// A fold whose response is recorded on the call's telemetry span.
pub struct Traced<Op: Operation, F> {
    inner: F,
    span: tracing::Span,
    record: fn(&tracing::Span, &Op::Response),
}

impl<Op, F> Fold<Op> for Traced<Op, F>
where
    Op: Operation,
    F: Fold<Op>,
{
    fn absorb(&mut self, event: &Op::Event) -> Result<(), ProviderError> {
        self.inner.absorb(event)
    }

    fn finish(self, end: Op::End, reply: Reply) -> Result<Op::Response, ProviderError> {
        let response = self.inner.finish(end, reply)?;
        (self.record)(&self.span, &response);
        Ok(response)
    }
}

/// Write the transport request id and the reply document onto a modality
/// response that did not name its own.
macro_rules! stamp_reply {
    ($response:ty) => {
        |response: &mut $response, reply: &Reply| {
            if response.provider_request_id.is_none() {
                response
                    .provider_request_id
                    .clone_from(&reply.provider_request_id);
            }
            if response.raw.is_null() {
                response.raw.clone_from(&reply.raw);
            }
        }
    };
}

modality_operation!(
    /// Embedding a batch of texts.
    Embedding {
        request: Vec<String>,
        response: crate::embeddings::EmbeddingResponse,
        telemetry: Embeddings,
        fold: Embedded,
        seed: |texts: &Vec<String>, call: &Call<'_>| Embedded::over(texts.clone(), call),
    }
);

modality_operation!(
    /// Embedding a batch of images from their encoded file bytes.
    ImageEmbedding {
        request: Vec<Vec<u8>>,
        response: crate::embeddings::ImageEmbeddingResponse,
        telemetry: Embeddings,
        fold: Embedded,
        seed: |images: &Vec<Vec<u8>>, call: &Call<'_>| Embedded::over(
            images.iter().map(|bytes| crate::embeddings::image_document(bytes)).collect(),
            call,
        ),
    }
);

modality_operation!(
    /// Ordering documents by relevance to a query.
    Rerank {
        request: RerankRequest,
        response: crate::rerank::RerankResponse,
        telemetry: Rerank,
        fold: Whole<Self>,
        seed: |_: &_, _: &Call<'_>| Whole::stamping(stamp_reply!(crate::rerank::RerankResponse)),
    }
);

modality_operation!(
    /// Transcribes audio using provider-specific request encoding.
    Transcription {
        request: crate::transcription::TranscriptionRequest,
        response: crate::transcription::TranscriptionResponse,
        telemetry: Transcription,
        fold: Whole<Self>,
        seed: |_: &_, _: &Call<'_>| Whole::stamping(stamp_reply!(crate::transcription::TranscriptionResponse)),
    }
);

#[cfg(feature = "image")]
modality_operation!(
    /// Generating an image.
    ImageGeneration {
        request: crate::image_generation::ImageGenerationRequest,
        response: crate::image_generation::ImageGenerationResponse,
        telemetry: ImageGeneration,
        fold: Whole<Self>,
        seed: |_: &_, _: &Call<'_>| Whole::stamping(stamp_reply!(crate::image_generation::ImageGenerationResponse)),
    }
);

#[cfg(feature = "audio")]
modality_operation!(
    /// Generating speech.
    AudioGeneration {
        request: crate::audio_generation::AudioGenerationRequest,
        response: crate::audio_generation::AudioGenerationResponse,
        telemetry: AudioGeneration,
        fold: Whole<Self>,
        seed: |_: &_, _: &Call<'_>| Whole::stamping(stamp_reply!(crate::audio_generation::AudioGenerationResponse)),
    }
);

/// Pairs a reply's vectors positionally with the request's documents.
/// Finishing rejects unequal vector and document counts, and widths that
/// contradict the width the caller declared.
pub struct Embedded {
    documents: Vec<String>,
    /// The provider and capabilities the reply is checked against.
    provider: String,
    capabilities: Capabilities,
}

impl Embedded {
    /// A fold that will zip its vectors onto `documents`, for the model
    /// `call` describes.
    pub fn over(documents: Vec<String>, call: &Call<'_>) -> Self {
        Self {
            documents,
            provider: call.wire.name.to_owned(),
            capabilities: call.wire.capabilities,
        }
    }

    /// The reply's vectors, paired with the inputs they belong to.
    fn zipped(self, vectors: Vec<Vector>) -> Result<Vec<Vector>, ProviderError> {
        self.capabilities.honour_declaration(
            &self.provider,
            vectors.iter().map(|vector| vector.vec.len()),
        )?;
        if vectors.len() != self.documents.len() {
            return Err(ProviderError::Response(format!(
                "provider returned {} embeddings for {} documents",
                vectors.len(),
                self.documents.len()
            )));
        }
        Ok(self
            .documents
            .into_iter()
            .zip(vectors)
            .map(|(document, vector)| Vector {
                document,
                vec: vector.vec,
            })
            .collect())
    }
}

impl Fold<Embedding> for Embedded {
    fn absorb(&mut self, event: &Infallible) -> Result<(), ProviderError> {
        match *event {}
    }

    fn finish(
        self,
        mut response: crate::embeddings::EmbeddingResponse,
        reply: Reply,
    ) -> Result<crate::embeddings::EmbeddingResponse, ProviderError> {
        response.embeddings = self.zipped(std::mem::take(&mut response.embeddings))?;
        stamp(&mut response.provider_request_id, &mut response.raw, &reply);
        Ok(response)
    }
}

impl Fold<ImageEmbedding> for Embedded {
    fn absorb(&mut self, event: &Infallible) -> Result<(), ProviderError> {
        match *event {}
    }

    fn finish(
        self,
        mut response: crate::embeddings::ImageEmbeddingResponse,
        reply: Reply,
    ) -> Result<crate::embeddings::ImageEmbeddingResponse, ProviderError> {
        response.embeddings = self.zipped(std::mem::take(&mut response.embeddings))?;
        stamp(&mut response.provider_request_id, &mut response.raw, &reply);
        Ok(response)
    }
}

/// Fill a response's request id and document from the reply when the
/// provider named neither.
fn stamp(provider_request_id: &mut Option<String>, raw: &mut serde_json::Value, reply: &Reply) {
    if provider_request_id.is_none() {
        provider_request_id.clone_from(&reply.provider_request_id);
    }
    if raw.is_null() {
        raw.clone_from(&reply.raw);
    }
}
