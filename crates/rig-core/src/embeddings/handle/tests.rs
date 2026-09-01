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

    async fn embed_texts_response(
        &self,
        texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        Ok(EmbeddingResponse::new(
            texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.0; self.inner.ndims],
                })
                .collect(),
            "probe",
        ))
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
        EmbeddingModel::embed_texts_response(&handle, vec!["d".to_owned()])
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

    async fn embed_texts_response(
        &self,
        _texts: impl IntoIterator<Item = String> + WasmCompatSend,
    ) -> Result<EmbeddingResponse, EmbeddingError> {
        Err(EmbeddingError::ResponseError("probe".to_owned()))
    }
}

#[test]
fn non_clone_models_erase() {
    fn assert_embedding_model<M: EmbeddingModel>() {}
    assert_embedding_model::<EmbeddingModelHandle>();
    let _ = || EmbeddingModelHandle::new(NonCloneModel);
}

/// The image twin: same invariant, same probe.
struct CloneCountingImageModel {
    clones: Arc<AtomicUsize>,
}

impl Clone for CloneCountingImageModel {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            clones: Arc::clone(&self.clones),
        }
    }
}

impl ImageEmbeddingModel for CloneCountingImageModel {
    fn max_documents(&self) -> usize {
        2
    }

    fn ndims(&self) -> usize {
        4
    }

    async fn embed_images_response(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> Result<ImageEmbeddingResponse, EmbeddingError> {
        Ok(ImageEmbeddingResponse::new(
            images
                .into_iter()
                .enumerate()
                .map(|(index, _)| Embedding {
                    document: format!("image-{index}"),
                    vec: vec![0.0; 4],
                })
                .collect(),
            "probe",
        ))
    }
}

#[tokio::test]
async fn image_erasure_never_clones_the_model() {
    let clones = Arc::new(AtomicUsize::new(0));
    let handle = ImageEmbeddingModelHandle::named(
        "img",
        CloneCountingImageModel {
            clones: Arc::clone(&clones),
        },
    );
    for _ in 0..3 {
        ImageEmbeddingModel::embed_images(&handle, vec![vec![1u8], vec![2u8]])
            .await
            .expect("embed");
        ImageEmbeddingModel::embed_image(&handle, &[3u8])
            .await
            .expect("embed one");
        ImageEmbeddingModel::embed_images_response(&handle.clone(), vec![vec![4u8]])
            .await
            .expect("embed response via clone");
    }
    assert_eq!(clones.load(Ordering::SeqCst), 0);
    assert_eq!(handle.ndims(), 4);
    assert_eq!(handle.max_documents(), 2);
    assert_eq!(handle.label(), Some("img"));
}
