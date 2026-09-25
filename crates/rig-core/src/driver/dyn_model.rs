//! A model erased to its operation: what a consumer stores when it holds
//! any model of one operation without naming its wire and transport. A
//! [`DynModel`] runs the same driver as the [`Model`] it was made from.

use std::fmt;
use std::sync::Arc;

use super::{Model, Step, Transport, completion_stream};
use crate::completion::CompletionRequest;
use crate::error::ProviderError;
use crate::observe::AdapterContext;
use crate::operation::Completion;
use crate::streaming::CompletionStream;
use crate::wasm_compat::{WasmBoxedFuture, WasmBoxedStream, WasmCompatSend, WasmCompatSync};
use crate::wire::{Mode, Operation, Wire};

/// Object-safe mirror of the calls a [`Model`] answers, with the wire and
/// transport fixed. Private: the only way to reach it is through
/// [`DynModel`], which re-exposes the public surface.
pub(crate) trait ErasedModel<Op: Operation>: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> &str;

    fn id(&self) -> Option<&str>;

    fn capabilities(&self) -> Op::Capabilities;

    fn fold(&self, request: &Op::Request, mode: Mode) -> Op::Fold;

    fn call(
        &self,
        request: Op::Request,
        observation: Option<AdapterContext>,
    ) -> WasmBoxedFuture<'_, Result<Op::Response, ProviderError>>;

    fn steps(
        &self,
        request: Op::Request,
        mode: Mode,
        observation: Option<AdapterContext>,
    ) -> Result<
        (
            tracing::Span,
            WasmBoxedStream<'static, Result<Step<Op>, ProviderError>>,
        ),
        ProviderError,
    >;
}

impl<W, T> ErasedModel<W::Op> for Model<W, T>
where
    W: Wire,
    T: Transport<W>,
{
    fn name(&self) -> &str {
        self.wire.name()
    }

    fn id(&self) -> Option<&str> {
        self.wire.id()
    }

    fn capabilities(&self) -> <W::Op as Operation>::Capabilities {
        self.wire.capabilities()
    }

    fn fold(
        &self,
        request: &<W::Op as Operation>::Request,
        mode: Mode,
    ) -> <W::Op as Operation>::Fold {
        <W::Op as Operation>::fold(request, &self.wire, mode)
    }

    fn call(
        &self,
        request: <W::Op as Operation>::Request,
        observation: Option<AdapterContext>,
    ) -> WasmBoxedFuture<'_, Result<<W::Op as Operation>::Response, ProviderError>> {
        Box::pin(self.unary(request, observation))
    }

    fn steps(
        &self,
        request: <W::Op as Operation>::Request,
        mode: Mode,
        observation: Option<AdapterContext>,
    ) -> Result<
        (
            tracing::Span,
            WasmBoxedStream<'static, Result<Step<W::Op>, ProviderError>>,
        ),
        ProviderError,
    > {
        Model::steps(self, request, mode, observation)
    }
}

/// A model of one operation with its wire and transport erased. Clones share
/// the model. Built with [`Model::erase`] or `From<Model<W, T>>`, so a
/// consumer takes `impl Into<DynModel<Op>>` and accepts either.
///
/// Every call runs the driver the concrete model runs: spans, request ids,
/// the operation's `accept` check and error enrichment are the same.
///
/// A model used by one consumer is passed as is; a model shared by several
/// is erased once and the handle is cloned. A model the bus serves is a
/// `ModelHandle` in the agent runtime instead: its calls are dispatched,
/// recorded and observed by the bus.
///
/// ```no_run
/// use rig_core::embeddings::EmbeddingsBuilder;
/// use rig_core::vector_store::in_memory_store::InMemoryVectorStore;
/// use rig_core::{Model, providers::openai::{self, OpenAI}};
///
/// # async fn example(http: rig_core::http_client::DynHttpClient) -> Result<(), Box<dyn std::error::Error>> {
/// let model = Model::new(OpenAI::from_env()?.embedding(openai::TEXT_EMBEDDING_3_SMALL, None), http).erase();
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(["a document".to_owned()])?
///     .build()
///     .await?;
/// let index = InMemoryVectorStore::from_documents(embeddings).index(model);
/// # let _ = index;
/// # Ok(())
/// # }
/// ```
pub struct DynModel<Op: Operation> {
    inner: Arc<dyn ErasedModel<Op>>,
}

impl<W, T> Model<W, T>
where
    W: Wire,
    T: Transport<W>,
{
    /// Erase this model to its operation.
    pub fn erase(self) -> DynModel<W::Op> {
        DynModel {
            inner: Arc::new(self),
        }
    }
}

impl<W, T> From<Model<W, T>> for DynModel<W::Op>
where
    W: Wire,
    T: Transport<W>,
{
    fn from(model: Model<W, T>) -> Self {
        model.erase()
    }
}

impl<Op: Operation> Clone for DynModel<Op> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<Op: Operation> fmt::Debug for DynModel<Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DynModel")
            .field("name", &self.name())
            .field("id", &self.id())
            .finish()
    }
}

impl<Op: Operation> DynModel<Op> {
    /// The wire's provider descriptor name (`"anthropic"`).
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// The model id the wire addresses, when the operation addresses one.
    pub fn id(&self) -> Option<&str> {
        self.inner.id()
    }

    /// What a runtime accounts for about this model.
    pub fn capabilities(&self) -> Op::Capabilities {
        self.inner.capabilities()
    }

    /// Send `request` and fold the whole reply into the operation's
    /// response; [`Model::call`] with the model erased.
    pub fn call(
        &self,
        request: Op::Request,
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + '_ {
        self.inner.call(request, None)
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: Op::Request,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + '_ {
        self.inner.call(request, Some(observation))
    }
}

impl DynModel<Completion> {
    /// Open a streamed completion; [`Model::stream`] with the model erased.
    pub fn stream(&self, request: CompletionRequest) -> Result<CompletionStream, ProviderError> {
        self.streamed(request, None)
    }

    /// [`Self::stream`], with the attempt observed under `observation`.
    pub fn stream_observed(
        &self,
        request: CompletionRequest,
        observation: AdapterContext,
    ) -> Result<CompletionStream, ProviderError> {
        self.streamed(request, Some(observation))
    }

    fn streamed(
        &self,
        request: CompletionRequest,
        observation: Option<AdapterContext>,
    ) -> Result<CompletionStream, ProviderError> {
        let fold = self.inner.fold(&request, Mode::Streaming);
        let (span, steps) = self.inner.steps(request, Mode::Streaming, observation)?;
        Ok(completion_stream(span, fold, steps))
    }
}

#[cfg(test)]
mod tests;
