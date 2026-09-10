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
    + crate::wasm_compat::WasmCompatSend {
        CompletionModel::completion(&self.inner, request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + crate::wasm_compat::WasmCompatSend {
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
    + crate::wasm_compat::WasmCompatSend {
        std::future::ready(Err(CompletionError::ProviderError(
            "compile-time probe".to_string(),
        )))
    }

    fn stream(
        &self,
        _request: CompletionRequest,
    ) -> impl Future<Output = Result<StreamingCompletionResponse, CompletionError>>
    + crate::wasm_compat::WasmCompatSend {
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
        (handle, named, via_arc)
    };
}
