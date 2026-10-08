//! A model erased to its operation: what a consumer stores when it holds
//! any model of one operation without naming its wire and transport. A
//! [`DynModel`] runs the same driver as the [`Model`] it was made from.

use std::fmt;
use std::sync::Arc;

use super::{Model, Transport};
use crate::error::ProviderError;
use crate::observe::AdapterContext;
use crate::streaming::Streamed;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use crate::wire::{Capabilities, Descriptor, Mode, Operation, Wire};

/// Object-safe mirror of the calls a [`Model`] answers, with the wire and
/// transport fixed. Private: the only way to reach it is through
/// [`DynModel`], which re-exposes the public surface.
pub(crate) trait ErasedModel<Op: Operation>: WasmCompatSend + WasmCompatSync {
    fn describe(&self) -> Descriptor<'_>;

    fn open(
        &self,
        request: Op::Request,
        mode: Mode,
        observation: Option<AdapterContext>,
    ) -> Result<Streamed<Op>, ProviderError>;
}

impl<W, T> ErasedModel<W::Op> for Model<W, T>
where
    W: Wire,
    T: Transport<W>,
{
    fn describe(&self) -> Descriptor<'_> {
        self.wire.describe()
    }

    fn open(
        &self,
        request: <W::Op as Operation>::Request,
        mode: Mode,
        observation: Option<AdapterContext>,
    ) -> Result<Streamed<W::Op>, ProviderError> {
        Model::open(self, request, mode, observation)
    }
}

/// A model of one operation with its wire and transport erased. Clones share
/// the model. Built with [`Model::erase`] or `From<Model<W, T>>`, so a
/// consumer takes `impl Into<DynModel<Op>>` and accepts either.
///
/// Every call runs the driver the concrete model runs: spans, request ids,
/// the operation's fold and error enrichment are the same.
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
/// let model = OpenAI::from_env()?.with_http(http).embedding(openai::TEXT_EMBEDDING_3_SMALL, None).erase();
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
        self.inner.describe().name
    }

    /// The model id the wire addresses, when the operation addresses one.
    pub fn id(&self) -> Option<&str> {
        self.inner.describe().model
    }

    /// The catalog facts of the model the wire addresses: the spec it was
    /// connected to, else the catalog's entry for its id. `None` for an
    /// operation that addresses no model, or a model the catalog does not
    /// list.
    pub fn spec(&self) -> Option<&crate::catalog::ModelSpec> {
        self.inner.describe().spec()
    }

    /// What a runtime accounts for about this model.
    pub fn capabilities(&self) -> Capabilities {
        self.inner.describe().capabilities
    }

    /// Send `request` and fold the whole reply into the operation's
    /// response; [`Model::call`] with the model erased. The call owns what
    /// it sends, so it can be spawned.
    pub fn call(
        &self,
        request: impl Into<Op::Request>,
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + 'static {
        self.finished(request.into(), None)
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: impl Into<Op::Request>,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + 'static {
        self.finished(request.into(), Some(observation))
    }

    /// The call opens when first polled, as [`Model::call`] does.
    fn finished(
        &self,
        request: Op::Request,
        observation: Option<AdapterContext>,
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + 'static {
        let inner = self.inner.clone();
        async move {
            inner
                .open(request, Mode::Unary, observation)?
                .finish()
                .await
        }
    }

    /// Open a streamed reply; [`Model::stream`] with the model erased.
    pub fn stream(&self, request: impl Into<Op::Request>) -> Result<Streamed<Op>, ProviderError> {
        self.inner.open(request.into(), Mode::Streaming, None)
    }

    /// [`Self::stream`], with the attempt observed under `observation`.
    pub fn stream_observed(
        &self,
        request: impl Into<Op::Request>,
        observation: AdapterContext,
    ) -> Result<Streamed<Op>, ProviderError> {
        self.inner
            .open(request.into(), Mode::Streaming, Some(observation))
    }
}

#[cfg(test)]
mod tests;
