use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use tower::Service;

use crate::completion::{CompletionError, CompletionModel, CompletionRequest, CompletionResponse};

pub struct CompletionService<M> {
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
