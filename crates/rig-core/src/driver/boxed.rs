//! A model erased to its operation: what a consumer stores when it holds
//! any model of one operation without naming its wire and transport. A
//! [`BoxedModel`] runs the same driver as the [`Model`] it was made from.
//!
//! ```no_run
//! use rig_core::wire::Wire as _;
//! use rig_core::{BoxedModel, Model, operation::Completion, providers::openai::OpenAI};
//!
//! # fn example(http: rig_core::http_client::BoxedHttpClient) -> Result<(), rig_core::client::EnvError> {
//! let model: BoxedModel<Completion> =
//!     OpenAI::from_env()?.completion("gpt-5.2").on(http).boxed();
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

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
/// [`BoxedModel`], which re-exposes the public surface.
pub(crate) trait ErasedModel<Op: Operation>: WasmCompatSend + WasmCompatSync {
    fn name(&self) -> &str;

    fn model(&self) -> Option<&str>;

    fn reasoning_issuer(&self, model: Option<&str>) -> Option<&str>;

    fn capabilities(&self) -> Op::Capabilities;

    fn call(
        &self,
        request: Op::Request,
        observation: Option<AdapterContext>,
    ) -> WasmBoxedFuture<'static, Result<Op::Response, ProviderError>>;

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

    fn model(&self) -> Option<&str> {
        self.wire.model()
    }

    fn reasoning_issuer(&self, model: Option<&str>) -> Option<&str> {
        self.wire.reasoning_issuer(model)
    }

    fn capabilities(&self) -> <W::Op as Operation>::Capabilities {
        self.wire.capabilities()
    }

    fn call(
        &self,
        request: <W::Op as Operation>::Request,
        observation: Option<AdapterContext>,
    ) -> WasmBoxedFuture<'static, Result<<W::Op as Operation>::Response, ProviderError>> {
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
/// the model. Built with [`Model::boxed`] or `From<Model<W, T>>`, so a
/// consumer takes `impl Into<BoxedModel<Op>>` and accepts either.
///
/// Every call runs the driver the concrete model runs: spans, request ids,
/// the operation's `accept` check and error enrichment are the same.
pub struct BoxedModel<Op: Operation> {
    inner: Arc<dyn ErasedModel<Op>>,
}

impl<W, T> Model<W, T>
where
    W: Wire,
    T: Transport<W>,
{
    /// Erase this model to its operation.
    pub fn boxed(self) -> BoxedModel<W::Op> {
        BoxedModel {
            inner: Arc::new(self),
        }
    }
}

impl<W, T> From<Model<W, T>> for BoxedModel<W::Op>
where
    W: Wire,
    T: Transport<W>,
{
    fn from(model: Model<W, T>) -> Self {
        model.boxed()
    }
}

impl<Op: Operation> Clone for BoxedModel<Op> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<Op: Operation> fmt::Debug for BoxedModel<Op> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoxedModel")
            .field("name", &self.name())
            .field("model", &self.model())
            .finish()
    }
}

impl<Op: Operation> BoxedModel<Op> {
    /// The wire's provider descriptor name (`"anthropic"`).
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// The model the wire addresses, when the operation addresses one.
    pub fn model(&self) -> Option<&str> {
        self.inner.model()
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
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + 'static {
        self.inner.call(request, None)
    }

    /// [`Self::call`], with the attempt observed under `observation`.
    pub fn call_observed(
        &self,
        request: Op::Request,
        observation: AdapterContext,
    ) -> impl Future<Output = Result<Op::Response, ProviderError>> + WasmCompatSend + 'static {
        self.inner.call(request, Some(observation))
    }
}

impl BoxedModel<Completion> {
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
        let issuer = self
            .inner
            .reasoning_issuer(request.model.as_deref().or(self.model()))
            .map(str::to_owned);
        let (span, steps) = self.inner.steps(request, Mode::Streaming, observation)?;
        Ok(completion_stream(span, self.name(), issuer, steps))
    }
}

#[cfg(test)]
mod tests;
