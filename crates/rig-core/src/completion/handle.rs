//! Runtime model handles: an erased [`CompletionModel`] plus its string identity.
//!
//! Provider authors implement [`CompletionModel`] as usual. [`ModelHandle`]
//! erases that implementation once, when it enters a long-lived runtime (an
//! agent, an ECS resource, a model registry), so that runtime can replace or
//! route models without changing its Rust type. Because completion responses are already normalized
//! at the provider boundary, the erasure is lossless: a handle is itself a
//! [`CompletionModel`] with the same unary and streaming behavior.
//!
//! [`CompletionModel::capabilities`] is captured **by value** at erasure time;
//! the handle never calls back into the provider for capability checks.

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};

use crate::{
    streaming::StreamingCompletionResponse,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

use super::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, ProviderCapabilities,
};

/// The string identity a specification, asset, or registry names a model by.
///
/// A [`ModelHandle`] is live process state and is never serialized; a
/// `ModelRef` is the serializable half — the label under which a runtime
/// resolves a handle (`ModelRef → ModelHandle`). It carries no provider
/// semantics: two refs are equal when their strings are equal.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelRef(Arc<str>);

// Transparent string (de)serialization without serde's `rc` feature.
impl Serialize for ModelRef {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ModelRef {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let label = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        Ok(Self(Arc::from(&*label)))
    }
}

impl ModelRef {
    /// Build a reference from any string-like value.
    pub fn new(label: impl Into<Arc<str>>) -> Self {
        Self(label.into())
    }

    /// The label as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::ops::Deref for ModelRef {
    type Target = str;

    fn deref(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelRef {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ModelRef {
    fn from(label: &str) -> Self {
        Self(Arc::from(label))
    }
}

impl From<String> for ModelRef {
    fn from(label: String) -> Self {
        Self(Arc::from(label))
    }
}

impl From<Arc<str>> for ModelRef {
    fn from(label: Arc<str>) -> Self {
        Self(label)
    }
}

impl From<ModelRef> for String {
    fn from(label: ModelRef) -> Self {
        label.0.to_string()
    }
}

impl PartialEq<str> for ModelRef {
    fn eq(&self, other: &str) -> bool {
        &*self.0 == other
    }
}

impl PartialEq<&str> for ModelRef {
    fn eq(&self, other: &&str) -> bool {
        &*self.0 == *other
    }
}

/// Private object-safe mirror of [`CompletionModel`], the same shape
/// `tower::BoxService` uses: the public trait stays generic (RPITIT futures),
/// this dyn-safe twin exists only so [`ModelHandle`] can store one vtable.
///
/// The `WasmCompat*` supertraits carry the cfg fork (no-op markers on browser
/// wasm), mirroring `ErasedTool` in `crate::tool` and `EmbeddingModelHandle` in
/// `crate::embeddings`. Capabilities are
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
    label: Option<ModelRef>,
    model: M,
}

/// A cloneable, opaque handle to live completion-model behavior.
///
/// The handle is the boundary between typed provider authoring and Rig's
/// runtimes (the futures agent driver, systems drivers, registries). It is intentionally not serializable:
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
/// use rig_core::completion::ModelHandle;
///
/// fn requires_serialize<T: serde::Serialize>() {}
/// requires_serialize::<ModelHandle>();
/// ```
///
/// ```compile_fail
/// use rig_core::completion::ModelHandle;
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

    /// Erase a typed completion model and attach a label.
    ///
    /// The label is the [`ModelRef`] a specification or registry would name
    /// this model by; on the handle itself it serves logs and routing
    /// diagnostics. It is not a stable provider identity and the handle is
    /// never serialized.
    pub fn named<M>(label: impl Into<ModelRef>, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::from_parts(Some(label.into()), model)
    }

    fn from_parts<M>(label: Option<ModelRef>, model: M) -> Self
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

    /// Returns the optional label attached to this handle, as a string.
    pub fn label(&self) -> Option<&str> {
        self.inner.label.as_deref()
    }

    /// Returns the optional label attached to this handle, as a [`ModelRef`].
    pub fn model_ref(&self) -> Option<&ModelRef> {
        self.inner.label.as_ref()
    }
}

/// A handle behaves exactly like the model it erased, with capabilities served
/// from the snapshot captured at erasure time.
///
/// It deliberately adds no request validation of its own. Drivers reach a
/// model through `CompletionRequestBuilder` (`send()` / `stream()`) and the
/// builder already runs
/// [`CompletionRequest::validate_message_content`]. Repeating it here would
/// scan the whole history a second time on every model call and buy nothing.
impl CompletionModel for ModelHandle {
    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse, CompletionError>>
    + crate::wasm_compat::WasmCompatSend {
        self.inner.model.completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + crate::wasm_compat::WasmCompatSend {
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
mod tests;
