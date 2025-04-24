use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use serde::{Deserialize, Serialize};
use tower::Service;

use crate::{completion::CompletionRequest, vector_store::VectorStoreIndex};

use super::ServiceError;

pub struct RagService<V, T> {
    vector_index: Arc<V>,
    num_results: usize,
    _phantom: PhantomData<T>,
}

impl<V, T> RagService<V, T>
where
    V: VectorStoreIndex,
{
    pub fn new(vector_index: V, num_results: usize) -> Self {
        Self {
            vector_index: Arc::new(vector_index),
            num_results,
            _phantom: PhantomData,
        }
    }
}

impl<V, T> Service<CompletionRequest> for RagService<V, T>
where
    V: VectorStoreIndex + 'static,
    T: Serialize + for<'a> Deserialize<'a> + Send,
{
    type Response = RagResult<T>;
    type Error = ServiceError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let vector_index = self.vector_index.clone();
        let num_results = self.num_results;

        Box::pin(async move {
            let Some(prompt) = req.chat_history.last().rag_text() else {
                return Err(ServiceError::required_option_not_exists("rag_text"));
            };

            let res = vector_index.top_n(&prompt, num_results).await?;

            Ok(res)
        })
    }
}

pub type RagResult<T> = Vec<(f64, String, T)>;
