//! Runtime model handles for the concrete agent facade.
//!
//! Provider authors continue to implement [`CompletionModel`] with typed unary
//! and streaming responses. [`ModelHandle`] erases that implementation once,
//! when it enters the high-level agent runtime, so an [`Agent`](super::Agent)
//! can replace or route models without changing its Rust type.
//!
//! Routing observes ten invariants: one selection occurs per `CallModel` step;
//! preparation reads that selection's capabilities; the same handle executes
//! the prepared request; in-flight work cannot be rebound; dropping a call
//! drops its retained future or stream; retries select anew; accepted tool calls
//! stay with their turn; history remains provider-neutral; concurrent runs own
//! independent routing snapshots; and replacing an agent clone has value
//! semantics. Provider-specific `additional_params` are not guaranteed to be
//! portable between providers and retain each provider's existing validation
//! and error behavior.

use std::{
    fmt,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use futures::{Stream, StreamExt};
use rig_core::{
    OneOrMany,
    completion::{CompletionError, CompletionModel, CompletionRequest, GetTokenUsage, Usage},
    message::AssistantContent,
    streaming::{StreamedAssistantContent, StreamingCompletionResponse},
    wasm_compat::WasmBoxedFuture,
};
use serde::{Deserialize, Serialize};

/// A provider-neutral final event on the high-level agent streaming surface.
///
/// Direct [`CompletionModel::stream`] calls still expose the provider's typed
/// streaming response. Agent streams normalize only the provider-final event so
/// callers can switch providers between model-call boundaries without changing
/// the stream's Rust type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct AgentStreamFinal {
    /// Token usage reported by the provider final response.
    pub usage: Usage,
    /// The provider final response serialized as JSON.
    pub raw_response: serde_json::Value,
}

impl GetTokenUsage for AgentStreamFinal {
    fn token_usage(&self) -> Usage {
        self.usage
    }
}

pub(crate) struct ModelCompletion {
    pub(crate) choice: OneOrMany<AssistantContent>,
    pub(crate) usage: Usage,
    pub(crate) message_id: Option<String>,
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type CompleteCallback = dyn Fn(CompletionRequest) -> WasmBoxedFuture<'static, Result<ModelCompletion, CompletionError>>
    + Send
    + Sync;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type CompleteCallback =
    dyn Fn(CompletionRequest) -> WasmBoxedFuture<'static, Result<ModelCompletion, CompletionError>>;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type StreamCallback = dyn Fn(CompletionRequest) -> WasmBoxedFuture<'static, Result<ModelStream, CompletionError>>
    + Send
    + Sync;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type StreamCallback =
    dyn Fn(CompletionRequest) -> WasmBoxedFuture<'static, Result<ModelStream, CompletionError>>;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type CapabilityCallback = dyn Fn() -> bool + Send + Sync;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type CapabilityCallback = dyn Fn() -> bool;

struct ModelDriver {
    complete: Arc<CompleteCallback>,
    open_stream: Arc<StreamCallback>,
    composes_native_output_with_tools: Arc<CapabilityCallback>,
    label: Option<String>,
}

/// A cloneable, opaque handle to live completion-model behavior.
///
/// The handle is the boundary between typed provider authoring and Rig's
/// concrete high-level agent facade. It is intentionally not serializable:
/// captured clients, credentials, transports, and callbacks are live process
/// state. Applications that need persistent model selection should serialize a
/// separate identifier and resolve it to a handle at runtime.
///
/// Cloning is cheap and shares the retained model through an [`Arc`]. Replacing
/// a handle on one cloned agent has value semantics and does not mutate other
/// agent clones. Each in-flight attempt owns its own handle clone, preserving
/// cancellation-by-drop without detached work.
///
/// The absence of serde implementations is intentional:
///
/// ```compile_fail
/// use rig_agent::ModelHandle;
///
/// fn requires_serialize<T: serde::Serialize>() {}
/// requires_serialize::<ModelHandle>();
/// ```
///
/// ```compile_fail
/// use rig_agent::ModelHandle;
///
/// fn requires_deserialize<T: for<'de> serde::Deserialize<'de>>() {}
/// requires_deserialize::<ModelHandle>();
/// ```
#[derive(Clone)]
pub struct ModelHandle {
    inner: Arc<ModelDriver>,
}

impl ModelHandle {
    /// Erase a typed completion model into a runtime model handle.
    pub fn new<M>(model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::from_parts(None, model)
    }

    /// Erase a typed completion model and attach a diagnostic label.
    ///
    /// Labels are for logs and routing diagnostics only. They are not stable
    /// provider identities and are not serialized.
    pub fn named<M>(label: impl Into<String>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::from_parts(Some(label.into()), model)
    }

    fn from_parts<M>(label: Option<String>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        let model = Arc::new(model);

        let complete_model = model.clone();
        let complete: Arc<CompleteCallback> = Arc::new(move |request| {
            let model = complete_model.clone();
            Box::pin(async move {
                let response = model.completion(request).await?;
                Ok(ModelCompletion {
                    choice: response.choice,
                    usage: response.usage,
                    message_id: response.message_id,
                })
            })
        });

        let stream_model = model.clone();
        let open_stream: Arc<StreamCallback> = Arc::new(move |request| {
            let model = stream_model.clone();
            Box::pin(async move {
                let stream = model.stream(request).await?;
                Ok(ModelStream::new(stream))
            })
        });

        let capability_model = model;
        let composes_native_output_with_tools: Arc<CapabilityCallback> =
            Arc::new(move || capability_model.composes_native_output_with_tools());

        Self {
            inner: Arc::new(ModelDriver {
                complete,
                open_stream,
                composes_native_output_with_tools,
                label,
            }),
        }
    }

    /// Returns the optional diagnostic label attached to this handle.
    pub fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    pub(crate) fn complete(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<ModelCompletion, CompletionError>> {
        (self.inner.complete)(request)
    }

    pub(crate) fn open_stream(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<ModelStream, CompletionError>> {
        (self.inner.open_stream)(request)
    }

    pub(crate) fn composes_native_output_with_tools(&self) -> bool {
        (self.inner.composes_native_output_with_tools)()
    }
}

impl fmt::Debug for ModelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelHandle")
            .field("label", &self.label())
            .finish_non_exhaustive()
    }
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
trait ErasedModelStream: Send {
    fn poll_next_erased(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<StreamedAssistantContent<AgentStreamFinal>, CompletionError>>>;

    fn choice(&self) -> &OneOrMany<AssistantContent>;

    fn message_id(&self) -> Option<&str>;
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
trait ErasedModelStream {
    fn poll_next_erased(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<StreamedAssistantContent<AgentStreamFinal>, CompletionError>>>;

    fn choice(&self) -> &OneOrMany<AssistantContent>;

    fn message_id(&self) -> Option<&str>;
}

struct TypedModelStream<R>
where
    R: Clone + Unpin + GetTokenUsage,
{
    inner: StreamingCompletionResponse<R>,
}

impl<R> ErasedModelStream for TypedModelStream<R>
where
    R: Clone + Unpin + rig_core::wasm_compat::WasmCompatSend + GetTokenUsage + Serialize + 'static,
{
    fn poll_next_erased(
        self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<Option<Result<StreamedAssistantContent<AgentStreamFinal>, CompletionError>>> {
        let this = self.get_mut();
        match this.inner.poll_next_unpin(context) {
            Poll::Ready(Some(Ok(item))) => Poll::Ready(Some(map_stream_item(item))),
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }

    fn choice(&self) -> &OneOrMany<AssistantContent> {
        &self.inner.choice
    }

    fn message_id(&self) -> Option<&str> {
        self.inner.message_id.as_deref()
    }
}

fn map_stream_item<R>(
    item: StreamedAssistantContent<R>,
) -> Result<StreamedAssistantContent<AgentStreamFinal>, CompletionError>
where
    R: Clone + Unpin + GetTokenUsage + Serialize,
{
    Ok(match item {
        StreamedAssistantContent::Text(text) => StreamedAssistantContent::Text(text),
        StreamedAssistantContent::ToolCall {
            tool_call,
            internal_call_id,
        } => StreamedAssistantContent::ToolCall {
            tool_call,
            internal_call_id,
        },
        StreamedAssistantContent::ToolCallDelta {
            id,
            internal_call_id,
            content,
        } => StreamedAssistantContent::ToolCallDelta {
            id,
            internal_call_id,
            content,
        },
        StreamedAssistantContent::Reasoning(reasoning) => {
            StreamedAssistantContent::Reasoning(reasoning)
        }
        StreamedAssistantContent::ReasoningDelta { id, reasoning } => {
            StreamedAssistantContent::ReasoningDelta { id, reasoning }
        }
        StreamedAssistantContent::Final(response) => {
            StreamedAssistantContent::Final(AgentStreamFinal {
                usage: response.token_usage(),
                raw_response: serde_json::to_value(response)?,
            })
        }
        StreamedAssistantContent::Unknown(value) => StreamedAssistantContent::Unknown(value),
    })
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type BoxedErasedModelStream = Pin<Box<dyn ErasedModelStream + Send>>;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type BoxedErasedModelStream = Pin<Box<dyn ErasedModelStream>>;

pub(crate) struct ModelStream {
    inner: BoxedErasedModelStream,
}

impl ModelStream {
    fn new<R>(stream: StreamingCompletionResponse<R>) -> Self
    where
        R: Clone
            + Unpin
            + rig_core::wasm_compat::WasmCompatSend
            + GetTokenUsage
            + Serialize
            + 'static,
    {
        Self {
            inner: Box::pin(TypedModelStream { inner: stream }),
        }
    }

    pub(crate) fn choice(&self) -> &OneOrMany<AssistantContent> {
        self.inner.as_ref().get_ref().choice()
    }

    pub(crate) fn message_id(&self) -> Option<&str> {
        self.inner.as_ref().get_ref().message_id()
    }
}

impl Stream for ModelStream {
    type Item = Result<StreamedAssistantContent<AgentStreamFinal>, CompletionError>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next_erased(context)
    }
}
