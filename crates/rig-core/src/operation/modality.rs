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
use crate::id::{ModelName, ResponseId};
use crate::response::{Reported, Response};
use crate::telemetry::{GenAiOperation, Recorded, SpanBuilder};
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
            output: $output:ty,
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
            type Event = Reported<$output>;
            type Response = Response<$output>;
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

            fn span(
                provider: &str,
                model: Option<&str>,
                telemetry: Self::Telemetry,
                _request: &Self::Request,
            ) -> tracing::Span {
                debug_assert!(!telemetry.is_completion());
                SpanBuilder::new(provider, model.unwrap_or_default(), telemetry).build()
            }

            fn recorded(response: &Self::Response) -> Option<Recorded<'_>> {
                Some(Recorded::from(&response.meta))
            }
        }
    };
}

modality_operation!(
    /// Embedding a batch of texts.
    Embedding {
        request: Vec<String>,
        output: Vec<Vector>,
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
        output: Vec<Vector>,
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
        output: Vec<crate::rerank::RerankResult>,
        capabilities: usize,
        telemetry: Rerank,
        name: "rerank",
        fold: Take<Vec<crate::rerank::RerankResult>>,
        seed: |_: &_| Take::default(),
    }
);

modality_operation!(
    /// Transcribes audio using provider-specific request encoding.
    Transcription {
        request: crate::transcription::TranscriptionRequest,
        output: String,
        capabilities: (),
        telemetry: Transcription,
        name: "transcription",
        fold: Take<String>,
        seed: |_: &_| Take::default(),
    }
);

#[cfg(feature = "image")]
modality_operation!(
    /// Generating an image.
    ImageGeneration {
        request: crate::image_generation::ImageGenerationRequest,
        output: Vec<u8>,
        capabilities: (),
        telemetry: ImageGeneration,
        name: "image_generation",
        fold: Take<Vec<u8>>,
        seed: |_: &_| Take::default(),
    }
);

#[cfg(feature = "audio")]
modality_operation!(
    /// Generating speech.
    AudioGeneration {
        request: crate::audio_generation::AudioGenerationRequest,
        output: Vec<u8>,
        capabilities: (),
        telemetry: AudioGeneration,
        name: "audio_generation",
        fold: Take<Vec<u8>>,
        seed: |_: &_| Take::default(),
    }
);

/// Accumulates vectors in reply order and pairs them positionally with request
/// documents. Finishing rejects missing replies or unequal vector/document counts.
/// Usage is summed; the model and response id come from the first reply.
#[derive(Default)]
pub struct Embedded {
    documents: Vec<String>,
    vectors: Vec<Vec<f64>>,
    /// The first reply's model and response id, once a reply arrived.
    first: Option<(Option<ModelName>, Option<ResponseId>)>,
    usage: crate::completion::Usage,
}

impl Embedded {
    /// A fold that will zip its vectors onto `documents`.
    pub fn over(documents: Vec<String>) -> Self {
        Self {
            documents,
            ..Self::default()
        }
    }
}

impl<Op> Fold<Op> for Embedded
where
    Op: Operation<Event = Reported<Vec<Vector>>, Response = Response<Vec<Vector>>>,
{
    fn absorb(&mut self, reply: Reported<Vec<Vector>>) -> Result<(), ProviderError> {
        self.vectors
            .extend(reply.output.into_iter().map(|vector| vector.vec));
        self.usage += reply.usage;
        self.first.get_or_insert((reply.model, reply.response_id));
        Ok(())
    }

    fn finish(self, reply: Reply) -> Result<Response<Vec<Vector>>, ProviderError> {
        if self.vectors.len() != self.documents.len() {
            return Err(ProviderError::Response(format!(
                "provider returned {} embeddings for {} documents",
                self.vectors.len(),
                self.documents.len()
            )));
        }
        let Some((model, response_id)) = self.first else {
            return Err(ProviderError::Response(
                "embedding reply carried no payload".to_owned(),
            ));
        };
        let output = self
            .documents
            .into_iter()
            .zip(self.vectors)
            .map(|(document, vec)| Vector { document, vec })
            .collect();
        Ok(Response {
            output,
            meta: reply.meta(model, response_id, self.usage),
        })
    }
}
