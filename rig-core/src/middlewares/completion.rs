use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use tower::Service;

use crate::completion::{CompletionError, CompletionModel, CompletionRequest, CompletionResponse};

/// A completion model as a Tower service.
///
/// This allows you to use an LLM model (or client) essentially anywhere you'd use a regular Tower layer, like in an Axum web service.
pub struct CompletionService<M> {
    /// The model itself.
    model: M,
}

impl<M> Service<CompletionRequest> for CompletionService<M>
where
    M: CompletionModel + 'static,
{
    type Response = CompletionResponse<M::Response>;
    type Error = CompletionError;

    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let model = self.model.clone();

        Box::pin(async move { model.completion(req).await })
    }
}
