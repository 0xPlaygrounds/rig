//! Typed views over the bus: one generic [`Handle<F>`], five aliases.
//!
//! A handle is a [`Dispatcher`] plus the key it is bound to, checked at bind
//! time against the handler's family. It is `Clone + Send + Sync + 'static`
//! on every target by construction and — deliberately — never serde: the
//! descriptor is the serde half. A scene stores the [`HandlerKey`] and its
//! [`HandlerDescriptor`] and re-binds with [`Dispatcher::handle`] at load.
//!
//! The stored descriptor is the bind-time snapshot. Because a runtime
//! registration can replace what serves a key (model swapping),
//! [`Handle::descriptor`] and [`ModelHandle::capabilities`] re-read the
//! dispatcher's table rather than the field; the field exists for the family
//! check and for hosts that serialize a handle's identity.
//!
//! Handles implement none of the impl-side traits: a consumer calls the
//! inherent methods below, which are dispatches.
//!
//! ```compile_fail
//! use rig_core::bus::ModelHandle;
//!
//! fn requires_serialize<T: serde::Serialize>() {}
//! requires_serialize::<ModelHandle>();
//! ```
//!
//! ```compile_fail
//! use rig_core::bus::ToolHandle;
//!
//! fn requires_deserialize<T: for<'de> serde::Deserialize<'de>>() {}
//! requires_deserialize::<ToolHandle>();
//! ```

use std::{
    fmt,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use futures::StreamExt;
use serde::de::DeserializeOwned;

use crate::{
    completion::{CompletionError, CompletionRequest, CompletionResponse, ProviderCapabilities},
    effect::{
        EffectId, EffectKind, EmbedInputs, EmbedModality, EmbedOutputs, Family, FamilyDescriptor,
        HandlerDescriptor, HandlerKey, MemoryOp, MemoryOutcome, Outcome, RetrieveQuery,
        RetrievedDocuments, family,
    },
    embeddings::{Embedding, EmbeddingResponse, ImageEmbeddingResponse},
    error::{ErrorKind, ErrorReport},
    id::ConversationId,
    message::Message,
    streaming::StreamingCompletionResponse,
    tool::{ToolContext, ToolResult},
    vector_store::request::{Filter, VectorSearchRequest},
};

use super::{Dispatcher, EffectStream, Pending};

/// A typed view over the bus for the family `F`.
#[derive(Clone)]
pub struct Handle<F: Family> {
    dispatcher: Dispatcher,
    descriptor: HandlerDescriptor,
    _family: PhantomData<fn() -> F>,
}

/// A completion model: `complete`, `stream`, `capabilities`.
pub type ModelHandle = Handle<family::Completion>;
/// A tool: `call`.
pub type ToolHandle = Handle<family::Tool>;
/// A conversation-memory backend: `load`, `append`, `clear`.
pub type MemoryHandle = Handle<family::Memory>;
/// A vector-store index: `top_n`, `top_n_ids`.
pub type IndexHandle = Handle<family::Retrieve>;
/// An embedding model; the modality is on the descriptor, not the type.
pub type EmbedHandle = Handle<family::Embed>;

impl<F: Family> fmt::Debug for Handle<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Handle")
            .field("family", &F::FAMILY)
            .field("key", &self.descriptor.key)
            .finish_non_exhaustive()
    }
}

impl<F: Family> Handle<F> {
    /// The key this handle dispatches to.
    pub fn key(&self) -> &HandlerKey {
        &self.descriptor.key
    }

    /// The descriptor *now*: re-read from the dispatcher's table, so a
    /// runtime replacement under the same key is visible. Falls back to the
    /// bind-time snapshot when nothing serves the key any more.
    pub fn descriptor(&self) -> HandlerDescriptor {
        self.dispatcher
            .descriptor(&self.descriptor.key)
            .unwrap_or_else(|| self.descriptor.clone())
    }

    /// The bind-time snapshot of the descriptor.
    pub fn bound_descriptor(&self) -> &HandlerDescriptor {
        &self.descriptor
    }

    /// Whether the bus behind this handle has closed.
    pub fn is_closed(&self) -> bool {
        self.dispatcher.is_closed()
    }

    /// The dispatcher this handle is a view over.
    pub fn dispatcher(&self) -> &Dispatcher {
        &self.dispatcher
    }

    fn dispatch(&self, kind: EffectKind) -> Pending {
        self.dispatcher.dispatch(&self.descriptor.key, kind)
    }
}

fn family_mismatch(
    key: &HandlerKey,
    wanted: crate::effect::EffectFamily,
    found: &FamilyDescriptor,
) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::HandlerUnavailable,
        format!(
            "handler `{key}` serves the {} family, not {wanted}",
            found.family()
        ),
    )
}

impl Dispatcher {
    /// Bind a typed view to `key`, checking the handler's family against
    /// `F` now rather than at first dispatch: asking for a [`ModelHandle`]
    /// at a tool key is `HandlerUnavailable` here.
    pub fn handle<F: Family>(&self, key: &HandlerKey) -> Result<Handle<F>, ErrorReport> {
        let descriptor = self
            .descriptor(key)
            .ok_or_else(|| super::dispatcher::handler_unavailable(key))?;
        if descriptor.family.family() != F::FAMILY {
            return Err(family_mismatch(key, F::FAMILY, &descriptor.family));
        }
        Ok(Handle {
            dispatcher: self.clone(),
            descriptor,
            _family: PhantomData,
        })
    }
}

/// A unary dispatch mapped to its typed answer: `Unpin`, executor-neutral,
/// cancelled by drop — the same value as [`Pending`] with the outcome
/// narrowed.
pub struct Typed<T> {
    pending: Pending,
    map: fn(Outcome) -> Result<T, ErrorReport>,
}

impl<T> Typed<T> {
    /// The dispatch's id.
    pub const fn id(&self) -> EffectId {
        self.pending.id()
    }
}

impl<T> fmt::Debug for Typed<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Typed").field("id", &self.id()).finish()
    }
}

impl<T> Unpin for Typed<T> {}

impl<T> Future for Typed<T> {
    type Output = Result<T, ErrorReport>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match Pin::new(&mut this.pending).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(report)) => Poll::Ready(Err(report)),
            Poll::Ready(Ok(outcome)) => Poll::Ready((this.map)(outcome)),
        }
    }
}

fn wrong_outcome(expected: &'static str, outcome: &Outcome) -> ErrorReport {
    ErrorReport::new(
        ErrorKind::Internal,
        format!(
            "expected a {expected} outcome, the handler answered {}",
            outcome.family()
        ),
    )
}

/// A completion dispatch in flight.
pub type Completion = Typed<CompletionResponse>;
/// A tool call in flight: the result and the context the tool published.
pub type ToolCall = Typed<(ToolResult, ToolContext)>;
/// A retrieval in flight.
pub type Retrieval<T> = Typed<Vec<(f64, String, T)>>;

impl ModelHandle {
    /// The capability snapshot the handler advertises now.
    pub fn capabilities(&self) -> ProviderCapabilities {
        match self.descriptor().family {
            FamilyDescriptor::Completion { capabilities, .. } => capabilities,
            FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => ProviderCapabilities::default(),
        }
    }

    /// The model's label as the handler advertises it now.
    pub fn model_ref(&self) -> crate::completion::ModelRef {
        match self.descriptor().family {
            FamilyDescriptor::Completion { model, .. } => model,
            FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => {
                crate::completion::ModelRef::new(self.key().as_str())
            }
        }
    }

    /// A unary completion.
    pub fn complete(&self, request: CompletionRequest) -> Completion {
        Typed {
            pending: self.dispatch(EffectKind::Completion {
                request,
                stream: false,
            }),
            map: |outcome| match outcome {
                Outcome::Completion(response) => Ok(response),
                other => Err(wrong_outcome("completion", &other)),
            },
        }
    }

    /// A streaming completion, wrapped back into a
    /// [`StreamingCompletionResponse`] over the B2 accumulator. Errors that
    /// cross the bus surface as [`CompletionError::Report`].
    pub fn stream(&self, request: CompletionRequest) -> StreamingCompletionResponse {
        let provider = self.model_ref().to_string();
        let stream: EffectStream = self.dispatcher.dispatch_stream(
            &self.descriptor.key,
            EffectKind::Completion {
                request,
                stream: true,
            },
        );
        wrap_stream(provider, stream)
    }
}

impl ToolHandle {
    /// The tool's name as advertised now.
    pub fn name(&self) -> String {
        match self.descriptor().family {
            FamilyDescriptor::Tool { name, .. } => name,
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Embed { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => self.key().to_string(),
        }
    }

    /// Call the tool with raw JSON `args` and a dispatch-scoped context.
    pub fn call(
        &self,
        name: impl Into<String>,
        args: impl Into<String>,
        context: ToolContext,
    ) -> ToolCall {
        Typed {
            pending: self.dispatch(EffectKind::ToolCall {
                name: name.into(),
                args: args.into(),
                context,
            }),
            map: |outcome| match outcome {
                Outcome::ToolResult { result, context } => Ok((result, context)),
                other => Err(wrong_outcome("tool result", &other)),
            },
        }
    }
}

impl MemoryHandle {
    /// Load a conversation's history.
    pub fn load(&self, conversation: ConversationId) -> Typed<Vec<Message>> {
        Typed {
            pending: self.dispatch(EffectKind::Memory {
                op: MemoryOp::Load { conversation },
            }),
            map: |outcome| match outcome {
                Outcome::Memory(MemoryOutcome::Loaded { messages }) => Ok(messages),
                other => Err(wrong_outcome("loaded memory", &other)),
            },
        }
    }

    /// Append messages to a conversation.
    pub fn append(&self, conversation: ConversationId, messages: Vec<Message>) -> Typed<()> {
        Typed {
            pending: self.dispatch(EffectKind::Memory {
                op: MemoryOp::Append {
                    conversation,
                    messages,
                },
            }),
            map: |outcome| match outcome {
                Outcome::Memory(MemoryOutcome::Appended) => Ok(()),
                other => Err(wrong_outcome("appended memory", &other)),
            },
        }
    }

    /// Clear a conversation.
    pub fn clear(&self, conversation: ConversationId) -> Typed<()> {
        Typed {
            pending: self.dispatch(EffectKind::Memory {
                op: MemoryOp::Clear { conversation },
            }),
            map: |outcome| match outcome {
                Outcome::Memory(MemoryOutcome::Cleared) => Ok(()),
                other => Err(wrong_outcome("cleared memory", &other)),
            },
        }
    }
}

impl IndexHandle {
    /// Scored documents, deserialized on this side of the bus: the wire
    /// carries JSON, the type parameter never crosses it.
    pub fn top_n<T: DeserializeOwned>(
        &self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> impl Future<Output = Result<Vec<(f64, String, T)>, ErrorReport>> + Unpin {
        let scored: Typed<Vec<(f64, String, serde_json::Value)>> = Typed {
            pending: self.dispatch(EffectKind::Retrieve {
                query: RetrieveQuery::TopN { req },
            }),
            map: |outcome| match outcome {
                Outcome::Documents(RetrievedDocuments::Scored(results)) => Ok(results),
                other => Err(wrong_outcome("scored documents", &other)),
            },
        };
        futures::future::FutureExt::map(scored, |result| {
            result.and_then(|results| {
                results
                    .into_iter()
                    .map(
                        |(score, id, document)| match serde_json::from_value::<T>(document) {
                            Ok(document) => Ok((score, id, document)),
                            Err(error) => Err(ErrorReport::new(
                                ErrorKind::Json,
                                format!("retrieved document `{id}` did not deserialize: {error}"),
                            )),
                        },
                    )
                    .collect()
            })
        })
    }

    /// Scored ids.
    pub fn top_n_ids(
        &self,
        req: VectorSearchRequest<Filter<serde_json::Value>>,
    ) -> Typed<Vec<(f64, String)>> {
        Typed {
            pending: self.dispatch(EffectKind::Retrieve {
                query: RetrieveQuery::TopNIds { req },
            }),
            map: |outcome| match outcome {
                Outcome::Documents(RetrievedDocuments::Ids(results)) => Ok(results),
                other => Err(wrong_outcome("scored ids", &other)),
            },
        }
    }
}

impl EmbedHandle {
    /// The modality the handler serves now.
    pub fn modality(&self) -> Option<EmbedModality> {
        match self.descriptor().family {
            FamilyDescriptor::Embed { modality, .. } => Some(modality),
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// The vector dimension the handler advertises now.
    pub fn ndims(&self) -> Option<usize> {
        match self.descriptor().family {
            FamilyDescriptor::Embed { dims, .. } => dims,
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// The largest batch the handler advertises now.
    pub fn max_documents(&self) -> Option<usize> {
        match self.descriptor().family {
            FamilyDescriptor::Embed { max_documents, .. } => Some(max_documents),
            FamilyDescriptor::Completion { .. }
            | FamilyDescriptor::Tool { .. }
            | FamilyDescriptor::Memory {}
            | FamilyDescriptor::Retrieve {}
            | FamilyDescriptor::Custom { .. } => None,
        }
    }

    /// Embed text documents.
    pub fn embed_texts(&self, texts: Vec<String>) -> Typed<EmbeddingResponse> {
        Typed {
            pending: self.dispatch(EffectKind::Embed {
                inputs: EmbedInputs::Texts(texts),
            }),
            map: |outcome| match outcome {
                Outcome::Embeddings(EmbedOutputs::Texts(response)) => Ok(response),
                other => Err(wrong_outcome("text embeddings", &other)),
            },
        }
    }

    /// Embed one text document.
    pub fn embed_text(
        &self,
        text: &str,
    ) -> impl Future<Output = Result<Embedding, ErrorReport>> + Unpin {
        futures::future::FutureExt::map(self.embed_texts(vec![text.to_owned()]), |result| {
            result.and_then(|mut response| {
                response.embeddings.pop().ok_or_else(|| {
                    ErrorReport::new(
                        ErrorKind::Response,
                        "embedding handler returned an empty response for embed_text",
                    )
                })
            })
        })
    }

    /// Embed image bytes.
    pub fn embed_images(&self, images: Vec<Vec<u8>>) -> Typed<ImageEmbeddingResponse> {
        Typed {
            pending: self.dispatch(EffectKind::Embed {
                inputs: EmbedInputs::Images(images),
            }),
            map: |outcome| match outcome {
                Outcome::Embeddings(EmbedOutputs::Images(response)) => Ok(response),
                other => Err(wrong_outcome("image embeddings", &other)),
            },
        }
    }
}

/// Wrap an [`EffectStream`] back into a [`StreamingCompletionResponse`]:
/// the B2 accumulator folds the events on this side of the bus, and errors
/// that crossed it surface as [`CompletionError::Report`].
pub fn wrap_stream(
    provider: impl Into<String>,
    stream: EffectStream,
) -> StreamingCompletionResponse {
    let inner = stream.map(|item| item.map_err(CompletionError::Report));
    StreamingCompletionResponse::stream(provider, Box::pin(inner))
}

// Every alias is `Clone + Send + Sync + 'static` on every target, by
// construction; a compiled assertion keeps it so under wasm32-unknown-unknown
// as well as natively.
const _: fn() = || {
    fn assert_view<T: Clone + Send + Sync + 'static>() {}
    assert_view::<ModelHandle>();
    assert_view::<ToolHandle>();
    assert_view::<MemoryHandle>();
    assert_view::<IndexHandle>();
    assert_view::<EmbedHandle>();
    fn assert_unpin<T: Unpin>() {}
    assert_unpin::<Completion>();
    assert_unpin::<ToolCall>();
};

#[cfg(test)]
mod tests;
