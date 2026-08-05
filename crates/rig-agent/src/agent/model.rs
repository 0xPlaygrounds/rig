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
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Private object-safe mirror of [`CompletionModel`], the same shape
/// `tower::BoxService` uses: the public trait stays generic (RPITIT futures),
/// this dyn-safe twin exists only so [`ModelHandle`] can store one vtable.
///
/// The `WasmCompat*` supertraits carry the cfg fork (no-op markers on browser
/// wasm), mirroring `ErasedTool` in `crate::tool`. Capabilities are
/// deliberately absent: they are construction-time data captured alongside the
/// erased model, not behavior to call back into.
trait ErasedModel: WasmCompatSend + WasmCompatSync {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'_, Result<CompletionResponse, CompletionError>>;

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'_, Result<StreamingCompletionResponse, CompletionError>>;
}

/// Every completion model erases; the borrowed futures delegate straight to
/// the RPITIT methods, so erasure adds one `Box::pin` per attempt and never
/// clones the model.
impl<M> ErasedModel for M
where
    M: CompletionModel + 'static,
{
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'_, Result<CompletionResponse, CompletionError>> {
        Box::pin(CompletionModel::completion(self, request))
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> WasmBoxedFuture<'_, Result<StreamingCompletionResponse, CompletionError>> {
        Box::pin(CompletionModel::stream(self, request))
    }
}

/// The handle's single allocation: snapshot data first, the unsized erased
/// model last, so `Arc<ModelDriver<M>>` unsize-coerces to
/// `Arc<ModelDriver<dyn ErasedModel>>` without a second box.
struct ModelDriver<M: ?Sized> {
    /// Capability snapshot taken at erasure time (see [`ProviderCapabilities`]).
    capabilities: ProviderCapabilities,
    label: Option<String>,
    model: M,
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
/// work never rebinds. The erased model is retained in a shared [`Arc`], so
/// each completion/stream attempt runs against the same instance: no per-call
/// clone of the model itself, and interior-mutable model state (counters,
/// rotating endpoints, local caches) persists across attempts.
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
    inner: Arc<ModelDriver<dyn ErasedModel>>,
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
        // Capture the capability snapshot once, at erasure time; the model is
        // consumed by value and never cloned again (pinned by the
        // `erasure_never_clones_the_model` test below).
        let capabilities = model.capabilities();
        Self {
            inner: Arc::new(ModelDriver {
                capabilities,
                label,
                model,
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
        self.inner.model.completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + rig_core::wasm_compat::WasmCompatSend {
        self.inner.model.stream(request)
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::test_utils::{MockCompletionModel, MockTurn};

    /// Wraps the mock model and counts every `Clone` of itself.
    struct CloneCountingModel {
        inner: MockCompletionModel,
        clones: Arc<AtomicUsize>,
    }

    impl Clone for CloneCountingModel {
        fn clone(&self) -> Self {
            self.clones.fetch_add(1, Ordering::SeqCst);
            Self {
                inner: self.inner.clone(),
                clones: Arc::clone(&self.clones),
            }
        }
    }

    impl CompletionModel for CloneCountingModel {
        fn completion(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CompletionError>>
        + rig_core::wasm_compat::WasmCompatSend {
            CompletionModel::completion(&self.inner, request)
        }

        fn stream(
            &self,
            request: CompletionRequest,
        ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
        + rig_core::wasm_compat::WasmCompatSend {
            CompletionModel::stream(&self.inner, request)
        }
    }

    /// Erasure consumes the model by value: no code path may ever clone it,
    /// no matter how many attempts run through the handle. This pins the
    /// shared-instance semantics structurally, not just in prose.
    #[tokio::test]
    async fn erasure_never_clones_the_model() {
        let clones = Arc::new(AtomicUsize::new(0));
        let model = CloneCountingModel {
            inner: MockCompletionModel::from_turns([
                MockTurn::text("one"),
                MockTurn::text("two"),
                MockTurn::text("three"),
            ]),
            clones: Arc::clone(&clones),
        };

        let handle = ModelHandle::new(model);
        let request = handle.completion_request("go").build();
        CompletionModel::completion(&handle, request.clone())
            .await
            .expect("first scripted turn");
        CompletionModel::completion(&handle, request.clone())
            .await
            .expect("second scripted turn");
        CompletionModel::completion(&handle, request)
            .await
            .expect("third scripted turn");

        let stream_clones = Arc::new(AtomicUsize::new(0));
        let stream_model = CloneCountingModel {
            inner: MockCompletionModel::from_stream_turns([
                vec![
                    crate::test_utils::MockStreamEvent::text("a"),
                    crate::test_utils::MockStreamEvent::final_response_with_default_usage(),
                ],
                vec![
                    crate::test_utils::MockStreamEvent::text("b"),
                    crate::test_utils::MockStreamEvent::final_response_with_default_usage(),
                ],
            ]),
            clones: Arc::clone(&stream_clones),
        };
        let stream_handle = ModelHandle::new(stream_model);
        let stream_request = stream_handle.completion_request("go").build();
        CompletionModel::stream(&stream_handle, stream_request.clone())
            .await
            .expect("first scripted stream turn");
        CompletionModel::stream(&stream_handle, stream_request)
            .await
            .expect("second scripted stream turn");

        assert_eq!(
            clones.load(Ordering::SeqCst),
            0,
            "erasure and unary attempts must never clone the model"
        );
        assert_eq!(
            stream_clones.load(Ordering::SeqCst),
            0,
            "erasure and streaming attempts must never clone the model"
        );
    }

    /// A model without any `Clone` impl at all must pass through every public
    /// erasure seam. The assertions are the bounds themselves — a regression
    /// is a compile error, which is the strongest form this check can take.
    struct NonCloneModel;

    impl CompletionModel for NonCloneModel {
        fn completion(
            &self,
            _request: CompletionRequest,
        ) -> impl Future<Output = Result<CompletionResponse, CompletionError>>
        + rig_core::wasm_compat::WasmCompatSend {
            std::future::ready(Err(CompletionError::ProviderError(
                "compile-time probe".to_string(),
            )))
        }

        fn stream(
            &self,
            _request: CompletionRequest,
        ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
        + rig_core::wasm_compat::WasmCompatSend {
            std::future::ready(Err(CompletionError::ProviderError(
                "compile-time probe".to_string(),
            )))
        }
    }

    #[test]
    fn traits() {
        fn assert_completion_model<M: CompletionModel>() {}

        assert_completion_model::<NonCloneModel>();
        // `Arc<M>` forwards the trait, so the documented "wrap it in an `Arc`
        // if needed" guidance holds for non-`Clone` models through the
        // generic builder path (`completion_request` gates on `Self: Clone`,
        // which `Arc<M>` always satisfies).
        assert_completion_model::<std::sync::Arc<NonCloneModel>>();

        // Construction through the public erasure seams type-checks without a
        // `Clone` impl; never awaited — the bounds are the test.
        let _ = || {
            let handle = ModelHandle::new(NonCloneModel);
            let named = ModelHandle::named("probe", NonCloneModel);
            let via_arc = std::sync::Arc::new(NonCloneModel).completion_request("go");
            let builder = crate::AgentBuilder::new(NonCloneModel);
            (handle, named, via_arc, builder)
        };
    }
}
