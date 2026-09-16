//! The unary modality operations: embeddings, reranking, transcription,
//! image generation, audio generation.
//!
//! Each is the same shape — one request, one reply document, one normalized
//! response carrying usage and identity — so they are declared once by
//! [`modality_operation!`] rather than five times by hand. The only
//! per-operation data is the request, the response, the error enum and the
//! canonical telemetry name.

use super::{One, Take};
use crate::embeddings::{Embedding as Vector, EmbeddingError};
use crate::telemetry::{ModalityOperation, ModalityResponseTelemetry, SpanCombinator};
use crate::wire::{Fold, Operation, Reply};

/// What a runtime accounts for on an embedding wire: the batch limit the
/// provider accepts, the dimensionality it returns, and the width the
/// caller declared when they named one.
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
    /// The width the caller *declared*, as opposed to the one
    /// [`Self::ndims`] resolved to.
    ///
    /// A distinct fact because the two answer different questions.
    /// `ndims` is what the model reports and a vector store sizes its index
    /// from: the caller's width when they named one, the provider's own
    /// table when they did not. This is only ever the caller's own claim,
    /// and `None` when they made none — which is why an unstated width can
    /// never mismatch, and a provider that disagrees with its own width
    /// table is not the caller's fault.
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

    /// Refuse a reply whose vectors are not the width the caller declared.
    ///
    /// This is the invariant behind [`EmbeddingError::MismatchedDimensions`],
    /// and it lives beside the capability rather than in [`Embedded`], the
    /// embeddings fold: the fold is seeded from the *request*, which for
    /// this operation is `Vec<String>`, so the declared width — a property
    /// of the wire, reached through [`Wire::capabilities`] — is not in
    /// scope there. The consumer impl that publishes `ndims()` off this
    /// same value is where both halves meet, so that is where it is called.
    ///
    /// The failure it catches is silent: the providers that ignore a
    /// `dimensions` field rather than rejecting it answer 200 with their
    /// native width, and `ndims()` goes on reporting the declared one until
    /// a vector store builds an index that cannot hold its own vectors.
    ///
    /// [`Wire::capabilities`]: crate::wire::Wire::capabilities
    pub(crate) fn honour_declaration(
        &self,
        provider: &str,
        widths: impl IntoIterator<Item = usize>,
    ) -> Result<(), EmbeddingError> {
        // Zero is rig's sentinel for an unknown width, never a claim about
        // one: a model absent from every table this build knows resolves to
        // it, and treating that as a declaration would fail every reply.
        let Some(requested) = self.declared.filter(|declared| *declared > 0) else {
            return Ok(());
        };
        let Some(returned) = widths.into_iter().find(|width| *width != requested) else {
            return Ok(());
        };
        Err(EmbeddingError::MismatchedDimensions {
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
            error: $error:ty,
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
            type Error = $error;
            type Capabilities = $capabilities;
            type Output = One<Self>;
            type Fold = $fold;
            type Telemetry = ModalityOperation;

            const NAME: &'static str = $name;

            fn is_terminal(_event: &Self::Event) -> bool {
                true
            }

            fn fold(request: &Self::Request) -> Self::Fold {
                #[allow(clippy::redundant_closure_call)]
                ($seed)(request)
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
        fold: Embedded,
        seed: |texts: &Vec<String>| Embedded::over(texts.clone()),
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
        error: crate::rerank::RerankError,
        capabilities: usize,
        telemetry: Rerank,
        name: "rerank",
        fold: Take<Self>,
        seed: |_: &_| Take::default(),
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
        error: crate::image_generation::ImageGenerationError,
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
        error: crate::audio_generation::AudioGenerationError,
        capabilities: (),
        telemetry: AudioGeneration,
        name: "audio_generation",
        fold: Take<Self>,
        seed: |_: &_| Take::default(),
    }
);

/// The fold of an embedding reply: the provider's vectors joined back onto
/// the request's own input.
///
/// [`Embedding::document`](crate::embeddings::Embedding) is the input the
/// vector belongs to, and it is *not* on the wire — Cohere echoes the texts,
/// Ollama and Voyage AI do not — so the fold carries the request's inputs
/// and zips them positionally, which is also the only place the
/// batch-length invariant every provider used to restate is now checked.
///
/// Replies accumulate: a provider that takes one item per request (Cohere
/// embeds one image per call) answers a batch with one reply each, in
/// request order.
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
    fn zipped(self) -> Result<(Vec<Vector>, Metadata, crate::completion::Usage), EmbeddingError> {
        if self.vectors.len() != self.documents.len() {
            return Err(EmbeddingError::ResponseError(format!(
                "provider returned {} embeddings for {} documents",
                self.vectors.len(),
                self.documents.len()
            )));
        }
        let Some(metadata) = self.metadata else {
            return Err(EmbeddingError::ResponseError(
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
        reply: crate::embeddings::EmbeddingResponse,
    ) -> Result<(), EmbeddingError> {
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

    fn finish(self, reply: Reply) -> Result<crate::embeddings::EmbeddingResponse, EmbeddingError> {
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
    ) -> Result<(), EmbeddingError> {
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
    ) -> Result<crate::embeddings::ImageEmbeddingResponse, EmbeddingError> {
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
