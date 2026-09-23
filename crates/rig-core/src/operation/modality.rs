//! Buffered embedding, reranking, transcription, image, and audio operations.
//!
//! ```
//! use rig_core::operation::EmbeddingCapabilities;
//!
//! let capabilities = EmbeddingCapabilities::new(32, 768).declaring(Some(768));
//! assert_eq!(capabilities.declared, Some(768));
//! ```

use super::{One, Take};
use crate::embeddings::Embedding as Vector;
use crate::error::ProviderError;
use crate::telemetry::{GenAiOperation, SpanBuilder, SpanCombinator};
use crate::wire::{Fold, Operation, Reply};

/// Embedding batch limit, resolved dimensions, and optional caller-declared width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EmbeddingCapabilities {
    /// The most documents the provider embeds in one request.
    pub max_documents: usize,
    /// The dimensionality of the returned vectors.
    pub ndims: usize,
    /// Width explicitly requested by the caller, not inferred from model metadata.
    /// `None` and zero disable declared-width validation.
    pub declared: Option<usize>,
}

impl EmbeddingCapabilities {
    /// The capability pair for a wire, with no width declared.
    pub const fn new(max_documents: usize, ndims: usize) -> Self {
        Self {
            max_documents,
            ndims,
            declared: None,
        }
    }

    /// Record the width the caller named, when they named one.
    ///
    /// A wire that still holds the caller's `Option<usize>` states it here;
    /// one that collapsed it to a resolved width at construction has
    /// nothing to state and leaves this `None`.
    pub const fn declaring(mut self, declared: Option<usize>) -> Self {
        self.declared = declared;
        self
    }

    /// Returns [`ProviderError::MismatchedDimensions`] for the first width
    /// differing from a positive caller declaration. Otherwise succeeds.
    pub(crate) fn honour_declaration(
        &self,
        provider: &str,
        widths: impl IntoIterator<Item = usize>,
    ) -> Result<(), ProviderError> {
        // Zero is rig's sentinel for an unknown width, never a claim about
        // one: a model absent from every table this build knows resolves to
        // it, and treating that as a declaration would fail every reply.
        let Some(requested) = self.declared.filter(|declared| *declared > 0) else {
            return Ok(());
        };
        let Some(returned) = widths.into_iter().find(|width| *width != requested) else {
            return Ok(());
        };
        Err(ProviderError::MismatchedDimensions {
            provider: provider.to_owned(),
            requested,
            returned,
        })
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
            capabilities: $capabilities:ty,
            telemetry: $telemetry:ident,
            name: $name:literal,
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
            type Capabilities = $capabilities;
            type Output = One<Self>;
            type Fold = $fold;
            type Telemetry = GenAiOperation;

            const NAME: &'static str = $name;

            fn is_terminal(_event: &Self::Event) -> bool {
                true
            }

            fn fold(request: &Self::Request) -> Self::Fold {
                #[allow(clippy::redundant_closure_call)]
                ($seed)(request)
            }

            fn telemetry(_streaming: bool) -> Self::Telemetry {
                GenAiOperation::$telemetry
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
                debug_assert!(!telemetry.is_completion());
                SpanBuilder::new(provider, model.unwrap_or_default(), telemetry).build()
            }

            fn record(span: &tracing::Span, response: &Self::Response) {
                span.record_response(
                    response.response_id.as_deref(),
                    response.model.as_deref(),
                    &response.usage,
                );
            }
        }
    };
}

modality_operation!(
    /// Embedding a batch of texts.
    Embedding {
        request: Vec<String>,
        response: crate::embeddings::EmbeddingResponse,
        capabilities: EmbeddingCapabilities,
        telemetry: Embeddings,
        name: "embedding",
        fold: Embedded,
        seed: |texts: &Vec<String>| Embedded::over(texts.clone()),
    }
);

modality_operation!(
    /// Embedding a batch of images from their encoded file bytes.
    ImageEmbedding {
        request: Vec<Vec<u8>>,
        response: crate::embeddings::ImageEmbeddingResponse,
        capabilities: EmbeddingCapabilities,
        telemetry: Embeddings,
        name: "image_embedding",
        fold: Embedded,
        seed: |images: &Vec<Vec<u8>>| Embedded::over(
            images.iter().map(|bytes| crate::embeddings::image_document(bytes)).collect(),
        ),
    }
);

modality_operation!(
    /// Ordering documents by relevance to a query.
    Rerank {
        request: RerankRequest,
        response: crate::rerank::RerankResponse,
        capabilities: usize,
        telemetry: Rerank,
        name: "rerank",
        fold: Take<Self>,
        seed: |_: &_| Take::default(),
    }
);

modality_operation!(
    /// Transcribes audio using provider-specific request encoding.
    Transcription {
        request: crate::transcription::TranscriptionRequest,
        response: crate::transcription::TranscriptionResponse,
        capabilities: (),
        telemetry: Transcription,
        name: "transcription",
        fold: Take<Self>,
        seed: |_: &_| Take::default(),
    }
);

#[cfg(feature = "image")]
modality_operation!(
    /// Generating an image.
    ImageGeneration {
        request: crate::image_generation::ImageGenerationRequest,
        response: crate::image_generation::ImageGenerationResponse,
        capabilities: (),
        telemetry: ImageGeneration,
        name: "image_generation",
        fold: Take<Self>,
        seed: |_: &_| Take::default(),
    }
);

#[cfg(feature = "audio")]
modality_operation!(
    /// Generating speech.
    AudioGeneration {
        request: crate::audio_generation::AudioGenerationRequest,
        response: crate::audio_generation::AudioGenerationResponse,
        capabilities: (),
        telemetry: AudioGeneration,
        name: "audio_generation",
        fold: Take<Self>,
        seed: |_: &_| Take::default(),
    }
);

/// Accumulates vectors in reply order and pairs them positionally with request
/// documents. Finishing rejects missing replies or unequal vector/document counts.
/// Usage is summed; other metadata comes from the first reply.
#[derive(Default)]
pub struct Embedded {
    documents: Vec<String>,
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
    /// A fold that will zip its vectors onto `documents`.
    pub fn over(documents: Vec<String>) -> Self {
        Self {
            documents,
            ..Self::default()
        }
    }

    fn absorb_parts(
        &mut self,
        vectors: impl IntoIterator<Item = Vector>,
        usage: crate::completion::Usage,
        metadata: Metadata,
    ) {
        self.vectors
            .extend(vectors.into_iter().map(|vector| vector.vec));
        self.usage += usage;
        self.metadata.get_or_insert(metadata);
    }

    /// The vectors, paired with the inputs they belong to.
    fn zipped(self) -> Result<(Vec<Vector>, Metadata, crate::completion::Usage), ProviderError> {
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
    fn absorb(&mut self, reply: crate::embeddings::EmbeddingResponse) -> Result<(), ProviderError> {
        self.absorb_parts(
            reply.embeddings,
            reply.usage,
            Metadata {
                provider: reply.provider,
                model: reply.model,
                response_id: reply.response_id,
                provider_request_id: reply.provider_request_id,
                raw: reply.raw,
            },
        );
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
        Embedding::stamp_reply(&mut response, reply);
        Ok(response)
    }
}

impl Fold<ImageEmbedding> for Embedded {
    fn absorb(
        &mut self,
        reply: crate::embeddings::ImageEmbeddingResponse,
    ) -> Result<(), ProviderError> {
        self.absorb_parts(
            reply.embeddings,
            reply.usage,
            Metadata {
                provider: reply.provider,
                model: reply.model,
                response_id: reply.response_id,
                provider_request_id: reply.provider_request_id,
                raw: reply.raw,
            },
        );
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
        ImageEmbedding::stamp_reply(&mut response, reply);
        Ok(response)
    }
}
