use std::{fmt::Debug, future::Future, marker::PhantomData, pin::Pin, task::Poll};

use tower::{Layer, Service};

use crate::{
    completion::{CompletionRequest, CompletionResponse},
    message::{AssistantContent, Text},
};

pub struct AwaitApprovalLayer<T> {
    integration: T,
}

impl<T> AwaitApprovalLayer<T>
where
    T: HumanInTheLoop + Clone,
{
    pub fn new(integration: T) -> Self {
        Self { integration }
    }

    pub fn with_predicate<R, D>(self, predicate: R) -> AwaitApprovalLayerWithPredicate<D, R, T>
    where
        D: Debug,
        R: Fn() -> Pin<Box<dyn Future<Output = bool> + Send>> + Clone + Send + 'static,
    {
        AwaitApprovalLayerWithPredicate {
            integration: self.integration,
            predicate,
            _t: PhantomData,
        }
    }
}

impl<S, T> Layer<S> for AwaitApprovalLayer<T>
where
    T: HumanInTheLoop + Clone,
{
    type Service = AwaitApprovalLayerService<S, T>;

    fn layer(&self, inner: S) -> Self::Service {
        AwaitApprovalLayerService::new(inner, self.integration.clone())
    }
}

pub struct AwaitApprovalLayerService<S, T> {
    inner: S,
    integration: T,
}

impl<S, T> AwaitApprovalLayerService<S, T>
where
    T: HumanInTheLoop,
{
    pub fn new(inner: S, integration: T) -> Self {
        Self { inner, integration }
    }
}

impl<S, T, Response> Service<CompletionRequest> for AwaitApprovalLayerService<S, T>
where
    S: Service<CompletionRequest, Response = CompletionResponse<Response>> + Clone + Send + 'static,
    S::Future: Send,
    Response: Clone + 'static + Send,
    T: HumanInTheLoop + Clone + Send + 'static,
{
    type Response = CompletionResponse<Response>;
    type Error = bool;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let mut inner = self.inner.clone();
        let await_approval_loop = self.integration.clone();

        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            let AssistantContent::Text(Text { text }) = res.choice.first() else {
                todo!("Handle error properly");
            };

            if await_approval_loop.send_message(&text).await.is_err() {
                todo!("Handle error properly");
            }

            let Ok(bool) = await_approval_loop.await_approval().await else {
                todo!("Handle error properly");
            };

            if bool {
                Ok(res)
            } else {
                todo!("Handle error properly - we should abort the pipeline here if the user wants to abort");
            }
        })
    }
}

pub struct AwaitApprovalLayerWithPredicate<P, R, T> {
    integration: T,
    predicate: R,
    _t: PhantomData<P>,
}

impl<D, R, S, T> Layer<S> for AwaitApprovalLayerWithPredicate<D, R, T>
where
    T: HumanInTheLoop + Clone,
    D: Debug,
    R: Fn(&D) -> Pin<Box<dyn Future<Output = bool> + Send>> + Clone + Send + 'static,
{
    type Service = AwaitApprovalLayerServiceWithPredicate<D, R, S, T>;

    fn layer(&self, inner: S) -> Self::Service {
        let predicate = self.predicate.clone();
        AwaitApprovalLayerServiceWithPredicate::new(
            inner,
            self.integration.clone(),
            predicate,
            self._t,
        )
    }
}

pub struct AwaitApprovalLayerServiceWithPredicate<D, R, S, T> {
    inner: S,
    integration: T,
    predicate: R,
    _t: PhantomData<D>,
}

impl<D, R, S, T> AwaitApprovalLayerServiceWithPredicate<D, R, S, T>
where
    T: HumanInTheLoop,
    R: Fn(&D) -> Pin<Box<dyn Future<Output = bool> + Send>> + Clone + Send + 'static,
    D: Debug,
{
    pub fn new(inner: S, integration: T, predicate: R, _t: PhantomData<D>) -> Self {
        Self {
            inner,
            integration,
            predicate,
            _t,
        }
    }
}

impl<D, S, T, Response, R> Service<CompletionRequest>
    for AwaitApprovalLayerServiceWithPredicate<D, R, S, T>
where
    R: Fn(&CompletionResponse<Response>) -> Pin<Box<dyn Future<Output = bool> + Send>>
        + Clone
        + Send
        + 'static,
    S: Service<CompletionRequest, Response = CompletionResponse<Response>> + Clone + Send + 'static,
    S::Future: Send,
    Response: Clone + 'static + Send,
    T: HumanInTheLoop + Clone + Send + 'static,
{
    type Response = CompletionResponse<Response>;
    type Error = bool;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let mut inner = self.inner.clone();
        let await_approval_loop = self.integration.clone();
        let predicate = self.predicate.clone();

        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            if predicate(&res).await {
                return Ok(res);
            }

            let AssistantContent::Text(Text { text }) = res.choice.first() else {
                todo!("Handle error properly");
            };

            if await_approval_loop.send_message(&text).await.is_err() {
                todo!("Handle error properly");
            }

            let Ok(bool) = await_approval_loop.await_approval().await else {
                todo!("Handle error properly");
            };

            if bool {
                Ok(res)
            } else {
                todo!("Handle error properly - we should abort the pipeline here if the user wants to abort");
            }
        })
    }
}

pub trait HumanInTheLoop {
    fn send_message(
        &self,
        res: &str,
    ) -> impl Future<Output = Result<(), Box<dyn std::error::Error>>> + Send;
    fn await_approval(
        &self,
    ) -> impl Future<Output = Result<bool, Box<dyn std::error::Error>>> + Send;
}

pub struct Stdout;

impl HumanInTheLoop for Stdout {
    async fn send_message(&self, res: &str) -> Result<(), Box<dyn std::error::Error>> {
        print!(
            "Current result: {res}

            Would you like to approve this step? [Y/n]"
        );

        Ok(())
    }

    async fn await_approval(&self) -> Result<bool, Box<dyn std::error::Error>> {
        let mut string = String::new();

        loop {
            std::io::stdin().read_line(&mut string).unwrap();

            match string.to_lowercase().trim() {
                "y" | "yes" => break Ok(true),
                "n" | "no" => break Ok(false),
                _ => println!("Please respond with 'y' or 'n'."),
            }
        }
    }
}
