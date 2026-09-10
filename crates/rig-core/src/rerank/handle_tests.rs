use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;

struct CloneCountingModel {
    clones: Arc<AtomicUsize>,
}

impl Clone for CloneCountingModel {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, Ordering::SeqCst);
        Self {
            clones: Arc::clone(&self.clones),
        }
    }
}

impl RerankModel for CloneCountingModel {
    fn max_documents(&self) -> usize {
        7
    }

    async fn rerank(
        &self,
        _query: &str,
        documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
        Ok(RerankResponse::new(
            documents
                .into_iter()
                .enumerate()
                .map(|(index, document)| RerankResult {
                    index,
                    document: Some(document),
                    relevance_score: 1.0,
                })
                .collect(),
            "probe",
        ))
    }
}

/// Erasure consumes the model by value: no code path may ever clone it,
/// no matter how many calls run through the handle.
#[tokio::test]
async fn erasure_never_clones_the_model() {
    let clones = Arc::new(AtomicUsize::new(0));
    let handle = RerankModelHandle::named(
        "probe",
        CloneCountingModel {
            clones: Arc::clone(&clones),
        },
    );
    for _ in 0..3 {
        RerankModel::rerank(&handle, "q", vec!["a".to_owned(), "b".to_owned()])
            .await
            .expect("rerank");
        RerankModel::rerank(&handle.clone(), "q", vec!["c".to_owned()])
            .await
            .expect("rerank via clone");
    }
    assert_eq!(clones.load(Ordering::SeqCst), 0);
    assert_eq!(handle.max_documents(), 7);
    assert_eq!(handle.label(), Some("probe"));
    assert_eq!(
        format!("{handle:?}"),
        "RerankModelHandle { label: Some(\"probe\"), max_documents: 7, .. }"
    );
}

struct NonCloneModel;

impl RerankModel for NonCloneModel {
    fn max_documents(&self) -> usize {
        1
    }

    async fn rerank(
        &self,
        _query: &str,
        _documents: Vec<String>,
    ) -> Result<RerankResponse, RerankError> {
        Err(RerankError::ResponseError("probe".to_owned()))
    }
}

#[test]
fn non_clone_models_erase() {
    fn assert_rerank_model<M: RerankModel>() {}
    assert_rerank_model::<RerankModelHandle>();
    let _ = || RerankModelHandle::new(NonCloneModel);
}
