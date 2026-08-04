//! Runtime model handles for the concrete agent facade.
//!
//! Provider authors implement [`CompletionModel`] as usual. [`ModelHandle`]
//! erases that implementation once, when it enters the high-level agent
//! runtime, so an [`Agent`](super::Agent) can replace or route models without
//! changing its Rust type. Because completion responses are already normalized
//! at the provider boundary, the erasure is lossless: a handle is itself a
//! [`CompletionModel`] with the same unary and streaming behavior.
//!
//! [`CompletionModel::capabilities`] is captured **by value** at erasure time;
//! the handle never calls back into the provider for capability checks.

use std::{fmt, sync::Arc};

use rig_core::{
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionResponse,
        ProviderCapabilities,
    },
    streaming::StreamingCompletionResponse,
    wasm_compat::WasmBoxedFuture,
};

// The `Send + Sync` bounds are dropped exactly where `rig-core`'s `WasmCompat*`
// markers go no-op — browser wasm.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type CompleteCallback = dyn Fn(CompletionRequest) -> WasmBoxedFuture<'static, Result<CompletionResponse, CompletionError>>
    + Send
    + Sync;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type CompleteCallback =
    dyn Fn(
        CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<CompletionResponse, CompletionError>>;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
type StreamCallback = dyn Fn(
        CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<StreamingCompletionResponse, CompletionError>>
    + Send
    + Sync;

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
type StreamCallback =
    dyn Fn(
        CompletionRequest,
    ) -> WasmBoxedFuture<'static, Result<StreamingCompletionResponse, CompletionError>>;

struct ModelDriver {
    complete: Box<CompleteCallback>,
    open_stream: Box<StreamCallback>,
    /// Capability snapshot taken at erasure time (see [`ProviderCapabilities`]).
    capabilities: ProviderCapabilities,
    label: Option<String>,
}

/// A cloneable, opaque handle to live completion-model behavior.
///
/// The handle is the boundary between typed provider authoring and Rig's
/// concrete high-level agent facade. It is intentionally not serializable:
/// captured clients, credentials, and transports are live process state.
/// Applications that need persistent model selection should serialize a
/// separate identifier and resolve it to a handle at runtime.
///
/// Cloning is cheap and shares the retained model through an [`Arc`]. Replacing
/// a handle on one cloned agent has value semantics and does not mutate other
/// agent clones; each in-flight attempt owns its own handle clone, so in-flight
/// work never rebinds. The erased model itself is cloned once per
/// completion/stream attempt, so a model whose `Clone` is expensive should be
/// wrapped in an [`Arc`] before erasure.
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
        // Capture the capability snapshot once, at erasure time.
        let capabilities = model.capabilities();
        let complete_model = model.clone();
        let complete: Box<CompleteCallback> = Box::new(move |request| {
            let model = complete_model.clone();
            Box::pin(async move { model.completion(request).await })
        });
        let open_stream: Box<StreamCallback> = Box::new(move |request| {
            let model = model.clone();
            Box::pin(async move { model.stream(request).await })
        });

        Self {
            inner: Arc::new(ModelDriver {
                complete,
                open_stream,
                capabilities,
                label,
            }),
        }
    }

    /// Returns the optional diagnostic label attached to this handle.
    pub fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }
}

/// A handle behaves exactly like the model it erased, with capabilities served
/// from the snapshot captured at erasure time.
impl CompletionModel for ModelHandle {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, CompletionError>>
    + rig_core::wasm_compat::WasmCompatSend {
        (self.inner.complete)(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + rig_core::wasm_compat::WasmCompatSend {
        (self.inner.open_stream)(request)
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities
    }
}

impl fmt::Debug for ModelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelHandle")
            .field("label", &self.label())
            .field("capabilities", &self.inner.capabilities)
            .finish_non_exhaustive()
    }
}
