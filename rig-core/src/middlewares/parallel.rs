use std::{future::Future, pin::Pin};

use tower::Service;

use crate::completion::CompletionRequest;

use super::ServiceError;

pub struct Stackable<A, B> {
    pub inner: A,
    pub outer: B,
}

impl<A, B> Stackable<A, B> {
    pub fn new(inner: A, outer: B) -> Self {
        Self { inner, outer }
    }

    pub fn take_values(self) -> (A, B) {
        (self.inner, self.outer)
    }
}

#[derive(Clone)]
pub struct ParallelService<S, T> {
    first_service: S,
    second_service: T,
}

impl<S, T> ParallelService<S, T>
where
    S: Service<CompletionRequest>,
    T: Service<CompletionRequest>,
{
    pub fn new(first_service: S, second_service: T) -> Self {
        Self {
            first_service,
            second_service,
        }
    }
}

impl<S, T> Service<CompletionRequest> for ParallelService<S, T>
where
    S: Service<CompletionRequest, Error = ServiceError> + Clone + Send + 'static,
    S::Future: Send,
    S::Response: Send + 'static,
    T: Service<CompletionRequest, Error = ServiceError> + Clone + Send + 'static,
    T::Future: Send,
    T::Response: Send + 'static,
{
    type Response = Stackable<S::Response, T::Response>;
    type Error = ServiceError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let mut first = self.first_service.clone();
        let mut second = self.second_service.clone();
        Box::pin(async move {
            let res1 = first.call(req.clone()).await?;
            let res2 = second.call(req.clone()).await?;

            let stackable = Stackable::new(res1, res2);

            Ok(stackable)
        })
    }
}

#[macro_export]
macro_rules! parallel_service {
    ($service1:tt, $service2:tt) => {
        $crate::pipeline::parallel::ParallelService::new($service1, $service2)
    };
    ($op1:tt $(, $ops:tt)*) => {
        $crate::pipeline::parallel::ParallelService::new(
            $service1,
            $crate::parallel_op!($($ops),*)
        )
    };
}
