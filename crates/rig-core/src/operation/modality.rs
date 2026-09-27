//! Buffered embedding, reranking, transcription, image, and audio operations.
//!
//! ```
//! use rig_core::wire::Capabilities;
//!
//! let capabilities = Capabilities::embedding(32, 768).declaring(Some(768));
//! assert_eq!(capabilities.declared, Some(768));
//! ```

use super::Take;
use crate::embeddings::Embedding as Vector;
use crate::error::ProviderError;
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Call, Capabilities, End, Fold, Mode, Operation, Ready, Reply};

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
            type Event = $response;
            type Response = $response;
            type Fold = Traced<Self, $fold>;

            fn is_terminal(_event: &Self::Event) -> bool {
                true
            }

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
                    mode: call.mode,
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

/// A fold whose reply is recorded on the call's telemetry span: a unary
/// call records its response, a stream records its one event.
pub struct Traced<Op: Operation, F> {
    inner: F,
    span: tracing::Span,
    mode: Mode,
    record: fn(&tracing::Span, &Op::Response),
}

impl<Op: Operation, F: Default> Default for Traced<Op, F> {
    /// A fold that records nothing: the one a decoder driven by hand needs.
    fn default() -> Self {
        Self {
            inner: F::default(),
            span: tracing::Span::none(),
            mode: Mode::Unary,
            record: |_, _| {},
        }
    }
}

impl<Op, F> Fold<Op> for Traced<Op, F>
where
    Op: Operation<Event = <Op as Operation>::Response>,
    F: Fold<Op>,
{
    fn absorb(&mut self, event: &Op::Event) -> Result<(), ProviderError> {
        if self.mode == Mode::Streaming {
            (self.record)(&self.span, event);
        }
        self.inner.absorb(event)
    }

    fn finish(self, reply: Reply) -> Result<Op::Response, ProviderError> {
        let response = self.inner.finish(reply)?;
        if self.mode == Mode::Unary {
            (self.record)(&self.span, &response);
        }
        Ok(response)
    }

    fn push(&mut self, item: Result<Op::Event, ProviderError>, ready: &mut Ready<Op>) {
        self.inner.push(item, ready);
    }

    fn unknown(&mut self, payload: crate::streaming::UnknownPayload, ready: &mut Ready<Op>) {
        self.inner.unknown(payload, ready);
    }

    fn settle(&mut self, ready: &mut Ready<Op>, end: Option<End>) {
        self.inner.settle(ready, end);
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
        fold: Take<Self>,
        seed: |_: &_, _: &Call<'_>| Take::stamping(stamp_reply!(crate::rerank::RerankResponse)),
    }
);

modality_operation!(
    /// Transcribes audio using provider-specific request encoding.
    Transcription {
        request: crate::transcription::TranscriptionRequest,
        response: crate::transcription::TranscriptionResponse,
        telemetry: Transcription,
        fold: Take<Self>,
        seed: |_: &_, _: &Call<'_>| Take::stamping(stamp_reply!(crate::transcription::TranscriptionResponse)),
    }
);

#[cfg(feature = "image")]
modality_operation!(
    /// Generating an image.
    ImageGeneration {
        request: crate::image_generation::ImageGenerationRequest,
        response: crate::image_generation::ImageGenerationResponse,
        telemetry: ImageGeneration,
        fold: Take<Self>,
        seed: |_: &_, _: &Call<'_>| Take::stamping(stamp_reply!(crate::image_generation::ImageGenerationResponse)),
    }
);

#[cfg(feature = "audio")]
modality_operation!(
    /// Generating speech.
    AudioGeneration {
        request: crate::audio_generation::AudioGenerationRequest,
        response: crate::audio_generation::AudioGenerationResponse,
        telemetry: AudioGeneration,
        fold: Take<Self>,
        seed: |_: &_, _: &Call<'_>| Take::stamping(stamp_reply!(crate::audio_generation::AudioGenerationResponse)),
    }
);

/// Accumulates vectors in reply order and pairs them positionally with request
/// documents. Finishing rejects missing replies, unequal vector/document
/// counts, and widths that contradict the width the caller declared. Usage is
/// summed; other metadata comes from the first reply. It sees each reply by
/// reference, so it copies the vectors it keeps.
#[derive(Default)]
pub struct Embedded {
    documents: Vec<String>,
    /// The provider and capabilities the replies are checked against.
    provider: String,
    capabilities: Capabilities,
    vectors: Vec<Vec<f64>>,
    /// The first reply's metadata; usage sums across replies.
    metadata: Option<Metadata>,
    usage: crate::completion::Usage,
}

/// What an embedding reply reports besides its vectors.
struct Metadata {
    provider: String,
    model: Option<String>,
    response_id: Option<String>,
    provider_request_id: Option<String>,
    raw: serde_json::Value,
}

impl Embedded {
    /// A fold that will zip its vectors onto `documents`, for the model
    /// `call` describes.
    pub fn over(documents: Vec<String>, call: &Call<'_>) -> Self {
        Self {
            documents,
            provider: call.wire.name.to_owned(),
            capabilities: call.wire.capabilities,
            ..Self::default()
        }
    }

    /// Keep one reply's vectors and usage, and its metadata when it is the
    /// first reply.
    fn absorb_parts(
        &mut self,
        vectors: &[Vector],
        usage: crate::completion::Usage,
        metadata: impl FnOnce() -> Metadata,
    ) {
        self.vectors
            .extend(vectors.iter().map(|vector| vector.vec.clone()));
        self.usage += usage;
        self.metadata.get_or_insert_with(metadata);
    }

    /// The vectors, paired with the inputs they belong to.
    fn zipped(self) -> Result<(Vec<Vector>, Metadata, crate::completion::Usage), ProviderError> {
        self.capabilities
            .honour_declaration(&self.provider, self.vectors.iter().map(Vec::len))?;
        if self.vectors.len() != self.documents.len() {
            return Err(ProviderError::Response(format!(
                "provider returned {} embeddings for {} documents",
                self.vectors.len(),
                self.documents.len()
            )));
        }
        let Some(metadata) = self.metadata else {
            return Err(ProviderError::Response(
                "embedding reply carried no payload".to_owned(),
            ));
        };
        let embeddings = self
            .documents
            .into_iter()
            .zip(self.vectors)
            .map(|(document, vec)| Vector { document, vec })
            .collect();
        Ok((embeddings, metadata, self.usage))
    }
}

impl Fold<Embedding> for Embedded {
    fn absorb(
        &mut self,
        reply: &crate::embeddings::EmbeddingResponse,
    ) -> Result<(), ProviderError> {
        self.absorb_parts(&reply.embeddings, reply.usage, || Metadata {
            provider: reply.provider.clone(),
            model: reply.model.clone(),
            response_id: reply.response_id.clone(),
            provider_request_id: reply.provider_request_id.clone(),
            raw: reply.raw.clone(),
        });
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<crate::embeddings::EmbeddingResponse, ProviderError> {
        let (embeddings, metadata, usage) = self.zipped()?;
        let mut response = crate::embeddings::EmbeddingResponse {
            embeddings,
            usage,
            provider: metadata.provider,
            model: metadata.model,
            response_id: metadata.response_id,
            provider_request_id: metadata.provider_request_id,
            raw: metadata.raw,
        };
        stamp(&mut response.provider_request_id, &mut response.raw, &reply);
        Ok(response)
    }
}

impl Fold<ImageEmbedding> for Embedded {
    fn absorb(
        &mut self,
        reply: &crate::embeddings::ImageEmbeddingResponse,
    ) -> Result<(), ProviderError> {
        self.absorb_parts(&reply.embeddings, reply.usage, || Metadata {
            provider: reply.provider.clone(),
            model: reply.model.clone(),
            response_id: reply.response_id.clone(),
            provider_request_id: reply.provider_request_id.clone(),
            raw: reply.raw.clone(),
        });
        Ok(())
    }

    fn finish(
        self,
        reply: Reply,
    ) -> Result<crate::embeddings::ImageEmbeddingResponse, ProviderError> {
        let (embeddings, metadata, usage) = self.zipped()?;
        let mut response = crate::embeddings::ImageEmbeddingResponse {
            embeddings,
            usage,
            provider: metadata.provider,
            model: metadata.model,
            response_id: metadata.response_id,
            provider_request_id: metadata.provider_request_id,
            raw: metadata.raw,
        };
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
