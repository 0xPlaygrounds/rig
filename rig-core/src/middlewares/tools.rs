use crate::{
    completion::{CompletionRequest, CompletionResponse},
    message::{AssistantContent, Message, ToolResultContent, UserContent},
    providers::{self},
    tool::{ToolSet, ToolSetError},
    OneOrMany,
};
use std::{future::Future, pin::Pin, sync::Arc, task::Poll};

use tower::{Layer, Service};

pub struct ToolLayer {
    tools: Arc<ToolSet>,
}

impl ToolLayer {
    pub fn new(tools: ToolSet) -> Self {
        Self {
            tools: Arc::new(tools),
        }
    }
}

impl<S> Layer<S> for ToolLayer {
    type Service = ToolLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ToolLayerService {
            inner,
            tools: Arc::clone(&self.tools),
        }
    }
}

pub struct ToolLayerService<S> {
    inner: S,
    tools: Arc<ToolSet>,
}

impl<S, T> Service<CompletionRequest> for ToolLayerService<S>
where
    S: Service<CompletionRequest, Response = CompletionResponse<T>> + Clone + Send + 'static,
    T: Send + 'static,
    S::Future: Send,
{
    type Response = (Vec<Message>, String, ToolResultContent);
    type Error = ToolSetError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let mut inner = self.inner.clone();
        let tools = self.tools.clone();
        let mut messages = req.chat_history.clone();

        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            let AssistantContent::ToolCall(tool_call) = res.choice.first() else {
                todo!("Handle error properly");
            };

            messages.push(Message::Assistant {
                content: OneOrMany::one(AssistantContent::ToolCall(tool_call.clone())),
            });

            let Ok(res) = tools
                .call(
                    &tool_call.function.name,
                    tool_call.function.arguments.to_string(),
                )
                .await
            else {
                todo!("Implement proper error handling");
            };

            Ok((messages, tool_call.id, ToolResultContent::text(res)))
        })
    }
}

pub struct ToolService {
    tools: Arc<ToolSet>,
}

type OpenAIResponse = CompletionResponse<providers::openai::CompletionResponse>;

impl Service<OpenAIResponse> for ToolService {
    type Response = Message;
    type Error = ToolSetError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: OpenAIResponse) -> Self::Future {
        let tools = self.tools.clone();

        Box::pin(async move {
            let crate::message::AssistantContent::ToolCall(tool_call) = req.choice.first() else {
                unimplemented!("handle error");
            };

            let Ok(res) = tools
                .call(
                    &tool_call.function.name,
                    tool_call.function.arguments.to_string(),
                )
                .await
            else {
                todo!("Implement proper error handling");
            };

            Ok(Message::User {
                content: OneOrMany::one(UserContent::tool_result(
                    tool_call.id,
                    OneOrMany::one(ToolResultContent::text(res)),
                )),
            })
        })
    }
}
