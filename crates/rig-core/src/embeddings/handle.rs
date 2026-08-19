//! A runtime handle that erases an [`EmbeddingModel`]'s concrete type.
//!
//! Provider authors implement [`EmbeddingModel`] as usual. [`EmbeddingModelHandle`]
//! erases that implementation once, when it enters a long-lived consumer such as
//! a vector store, so the store's Rust type no longer names the provider. The
//! handle is itself an [`EmbeddingModel`] with the same behavior — erasure is
//! lossless, because [`Embedding`] is already provider-neutral.
//!
//! This is the embedding twin of `rig_agent::ModelHandle`, and follows its
//! shape deliberately: a private object-safe mirror trait with a blanket impl,
//! one `Arc` allocation holding snapshot data plus the unsized erased model,
//! and construction-time data ([`EmbeddingModel::ndims`],
//! [`EmbeddingModel::max_documents`]) captured **by value** at erasure so the
//! handle never calls back into the provider for them.
//!
//! It is *not* a runtime-swapping mechanism. Swapping the embedding model
//! under a populated vector index changes the vector space and usually breaks
//! `ndims`; the handle exists for type ergonomics and dyn-storability
//! (heterogeneous collections of indexes, no provider name in every downstream
//! signature), and offers no way to replace the model it holds.

use std::{fmt, sync::Arc};

use crate::wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync};

use super::{Embedding, EmbeddingError, EmbeddingModel, EmbeddingResponse};

/// Private object-safe mirror of [`EmbeddingModel`]: the public trait stays
/// generic (RPITIT futures, `impl IntoIterator` arguments); this dyn-safe twin
/// exists only so [`EmbeddingModelHandle`] can store one vtable.
///
/// The `WasmCompat*` supertraits carry the cfg fork (no-op markers on browser
/// wasm). `ndims` and `max_documents` are deliberately absent: they are
/// construction-time data captured alongside the erased model.
trait ErasedEmbeddingModel: WasmCompatSend + WasmCompatSync {
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> WasmBoxedFuture<'_, Result<Vec<Embedding>, EmbeddingError>>;

    fn embed_texts_with_usage(
        &self,
        texts: Vec<String>,
    ) -> WasmBoxedFuture<'_, Result<EmbeddingResponse, EmbeddingError>>;
}

/// Every embedding model erases; the borrowed futures delegate straight to the
/// RPITIT methods, so erasure adds one `Box::pin` per call and never clones
/// the model.
impl<M> ErasedEmbeddingModel for M
where
    M: EmbeddingModel + 'static,
{
    fn embed_texts(
        &self,
        texts: Vec<String>,
    ) -> WasmBoxedFuture<'_, Result<Vec<Embedding>, EmbeddingError>> {
        Box::pin(EmbeddingModel::embed_texts(self, texts))
    }

    fn embed_texts_with_usage(
        &self,
        texts: Vec<String>,
    ) -> WasmBoxedFuture<'_, Result<EmbeddingResponse, EmbeddingError>> {
        Box::pin(EmbeddingModel::embed_texts_with_usage(self, texts))
    }
}

/// The handle's single allocation: snapshot data first, the unsized erased
/// model last, so `Arc<EmbeddingDriver<M>>` unsize-coerces to
/// `Arc<EmbeddingDriver<dyn ErasedEmbeddingModel>>` without a second box.
struct EmbeddingDriver<M: ?Sized> {
    ndims: usize,
    max_documents: usize,
    label: Option<String>,
    model: M,
}

/// A cloneable, opaque handle to live embedding-model behavior.
///
/// Cloning is cheap and shares the retained model through an [`Arc`]; every
/// call runs against the same instance, so interior-mutable model state
/// persists and the model itself is never cloned. The handle is intentionally
/// not serializable: captured clients and credentials are live process state.
#[derive(Clone)]
pub struct EmbeddingModelHandle {
    inner: Arc<EmbeddingDriver<dyn ErasedEmbeddingModel>>,
}

impl EmbeddingModelHandle {
    /// Erase a typed embedding model into a runtime handle.
    pub fn new<M>(model: M) -> Self
    where
        M: EmbeddingModel + 'static,
    {
        Self::from_parts(None, model)
    }

    /// Erase a typed embedding model and attach a diagnostic label.
    ///
    /// Labels are for logs and diagnostics only; they are not stable provider
    /// identities and are not serialized.
    pub fn named<M>(label: impl Into<String>, model: M) -> Self
    where
        M: EmbeddingModel + 'static,
    {
        Self::from_parts(Some(label.into()), model)
    }

    fn from_parts<M>(label: Option<String>, model: M) -> Self
    where
        M: EmbeddingModel + 'static,
    {
        // Capture the snapshot once, at erasure time; the model is consumed
        // by value and never cloned again (pinned by the
        // `erasure_never_clones_the_model` test below).
        let ndims = model.ndims();
        let max_documents = model.max_documents();
        Self {
            inner: Arc::new(EmbeddingDriver {
                ndims,
                max_documents,
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

/// A handle behaves exactly like the model it erased, with `ndims` and
/// `max_documents` served from the snapshot captured at erasure time.
impl EmbeddingModel for EmbeddingModelHandle {
    fn max_documents(&self) -> usize {
        self.inner.max_documents
    }

    fn ndims(&self) -> usize {
        self.inner.ndims
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<Vec<Embedding>, EmbeddingError>> + WasmCompatSend
    {
        self.inner.model.embed_texts(texts.into_iter().collect())
    }

    fn embed_texts_with_usage(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> impl std::future::Future<Output = Result<EmbeddingResponse, EmbeddingError>> + WasmCompatSend
    {
        self.inner
            .model
            .embed_texts_with_usage(texts.into_iter().collect())
    }
}

impl fmt::Debug for EmbeddingModelHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EmbeddingModelHandle")
            .field("label", &self.label())
            .field("ndims", &self.inner.ndims)
            .field("max_documents", &self.inner.max_documents)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A minimal model that counts every `Clone` of itself.
    #[derive(Clone)]
    struct CountingInner {
        ndims: usize,
    }

    struct CloneCountingModel {
        inner: CountingInner,
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

    impl EmbeddingModel for CloneCountingModel {
        fn max_documents(&self) -> usize {
            4
        }

        fn ndims(&self) -> usize {
            self.inner.ndims
        }

        async fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + WasmCompatSend,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            Ok(texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0; self.inner.ndims],
                })
                .collect())
        }
    }

    /// Erasure consumes the model by value: no code path may ever clone it,
    /// no matter how many calls run through the handle. This pins the
    /// shared-instance semantics structurally, not just in prose.
    #[tokio::test]
    async fn erasure_never_clones_the_model() {
        let clones = Arc::new(AtomicUsize::new(0));
        let handle = EmbeddingModelHandle::new(CloneCountingModel {
            inner: CountingInner { ndims: 3 },
            clones: Arc::clone(&clones),
        });

        for _ in 0..3 {
            EmbeddingModel::embed_texts(&handle, vec!["a".to_owned(), "b".to_owned()])
                .await
                .expect("embed");
            EmbeddingModel::embed_text(&handle, "c")
                .await
                .expect("embed one");
            EmbeddingModel::embed_texts_with_usage(&handle, vec!["d".to_owned()])
                .await
                .expect("embed with usage");
        }
        let second = handle.clone();
        EmbeddingModel::embed_text(&second, "e")
            .await
            .expect("embed via clone");

        assert_eq!(
            clones.load(Ordering::SeqCst),
            0,
            "erasure and calls through the handle must never clone the model"
        );
    }

    /// `ndims` and `max_documents` are snapshots, not callbacks.
    #[test]
    fn snapshot_is_captured_by_value() {
        let handle = EmbeddingModelHandle::named(
            "probe",
            CloneCountingModel {
                inner: CountingInner { ndims: 5 },
                clones: Arc::new(AtomicUsize::new(0)),
            },
        );
        assert_eq!(handle.ndims(), 5);
        assert_eq!(handle.max_documents(), 4);
        assert_eq!(handle.label(), Some("probe"));
        assert_eq!(
            format!("{handle:?}"),
            "EmbeddingModelHandle { label: Some(\"probe\"), ndims: 5, max_documents: 4, .. }"
        );
    }

    /// A model without any `Clone` impl passes through the erasure seam; the
    /// bound is the test.
    struct NonCloneModel;

    impl EmbeddingModel for NonCloneModel {
        fn max_documents(&self) -> usize {
            1
        }

        fn ndims(&self) -> usize {
            1
        }

        async fn embed_texts(
            &self,
            _texts: impl IntoIterator<Item = String> + WasmCompatSend,
        ) -> Result<Vec<Embedding>, EmbeddingError> {
            Err(EmbeddingError::ResponseError("probe".to_owned()))
        }
    }

    #[test]
    fn non_clone_models_erase() {
        fn assert_embedding_model<M: EmbeddingModel>() {}
        assert_embedding_model::<EmbeddingModelHandle>();
        let _ = || EmbeddingModelHandle::new(NonCloneModel);
    }
}
