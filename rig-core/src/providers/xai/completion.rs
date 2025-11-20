// ================================================================
//! xAI Completion Integration
//! From [xAI Reference](https://docs.x.ai/docs/api-reference#chat-completions)
// ================================================================

use crate::{
    completion::{self, CompletionError},
    http_client::HttpClientExt,
    models,
    providers::openai::Message,
};

use super::client::{Client, xai_api_types::ApiResponse};
use crate::completion::CompletionRequest;
use crate::providers::openai;
use crate::streaming::StreamingCompletionResponse;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use tracing::{Instrument, info_span};
use xai_api_types::{CompletionResponse, ToolDefinition};

models! {
    #[allow(non_camel_case_types)]
    /// xAI completion models as of 2025-06-04
    pub enum CompletionModels {
        Grok2_1212 => "grok-2-1212",
        Grok2Vision_1212 => "grok-2-vision-1212",
        Grok3 => "grok-3",
        Grok3Fast => "grok-3-fast",
        Grok3Mini => "grok-3-mini",
        Grok3MiniFast => "grok-3-mini-fast",
        Grok2Image_1212 => "grok-2-image-1212",
        Grok4 => "grok-4-0709",
    }
}
pub use CompletionModels::*;

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct XAICompletionRequest {
    model: String,
    pub messages: Vec<Message>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ToolDefinition>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    tool_choice: Option<crate::providers::openrouter::ToolChoice>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub additional_params: Option<serde_json::Value>,
}

impl TryFrom<(&str, CompletionRequest)> for XAICompletionRequest {
    type Error = CompletionError;

    fn try_from((model, req): (&str, CompletionRequest)) -> Result<Self, Self::Error> {
        let mut full_history: Vec<Message> = match &req.preamble {
            Some(preamble) => vec![Message::system(preamble)],
            None => vec![],
        };
        if let Some(docs) = req.normalized_documents() {
            let docs: Vec<Message> = docs.try_into()?;
            full_history.extend(docs);
        }

        let chat_history: Vec<Message> = req
            .chat_history
            .clone()
            .into_iter()
            .map(|message| message.try_into())
            .collect::<Result<Vec<Vec<Message>>, _>>()?
            .into_iter()
            .flatten()
            .collect();

        full_history.extend(chat_history);

        let tool_choice = req
            .tool_choice
            .clone()
            .map(crate::providers::openrouter::ToolChoice::try_from)
            .transpose()?;

        Ok(Self {
            model: model.to_string(),
            messages: full_history,
            temperature: req.temperature,
            tools: req
                .tools
                .clone()
                .into_iter()
                .map(ToolDefinition::from)
                .collect::<Vec<_>>(),
            tool_choice,
            additional_params: req.additional_params,
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: CompletionModels,
}

impl<T> CompletionModel<T> {
    pub fn new(client: Client<T>, model: CompletionModels) -> Self {
        Self { client, model }
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt + Clone + Default + std::fmt::Debug + Send + 'static,
{
    type Response = CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;
    type Models = CompletionModels;

    fn make(client: &Self::Client, model: impl Into<Self::Models>) -> Self {
        Self::new(client.clone(), model.into())
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn completion(
        &self,
        completion_request: completion::CompletionRequest,
    ) -> Result<completion::CompletionResponse<CompletionResponse>, CompletionError> {
        let preamble = completion_request.preamble.clone();
        let request =
            XAICompletionRequest::try_from((self.model.to_string().as_ref(), completion_request))?;
        let request_messages_json_str = serde_json::to_string(&request.messages).unwrap();

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "xai",
                gen_ai.request.model = <CompletionModels as Into<&str>>::into(self.model),
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = &request_messages_json_str,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::debug!("xAI completion request: {request_messages_json_str}");

        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/v1/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| CompletionError::HttpError(e.into()))?;

        async move {
            let response = self.client.http_client().send::<_, Bytes>(req).await?;
            let status = response.status();
            let response_body = response.into_body().into_future().await?.to_vec();

            if status.is_success() {
                match serde_json::from_slice::<ApiResponse<CompletionResponse>>(&response_body)? {
                    ApiResponse::Ok(completion) => completion.try_into(),
                    ApiResponse::Error(error) => {
                        Err(CompletionError::ProviderError(error.message()))
                    }
                }
            } else {
                Err(CompletionError::ProviderError(
                    String::from_utf8_lossy(&response_body).to_string(),
                ))
            }
        }
        .instrument(span)
        .await
    }

    #[cfg_attr(feature = "worker", worker::send)]
    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
        CompletionModel::stream(self, request).await
    }
}

pub mod xai_api_types {
    use serde::{Deserialize, Serialize};

    use crate::OneOrMany;
    use crate::completion::{self, CompletionError};
    use crate::providers::openai::{AssistantContent, Message};

    impl TryFrom<CompletionResponse> for completion::CompletionResponse<CompletionResponse> {
        type Error = CompletionError;

        fn try_from(response: CompletionResponse) -> Result<Self, Self::Error> {
            let choice = response.choices.first().ok_or_else(|| {
                CompletionError::ResponseError("Response contained no choices".to_owned())
            })?;
            let content = match &choice.message {
                Message::Assistant {
                    content,
                    tool_calls,
                    ..
                } => {
                    let mut content = content
                        .iter()
                        .map(|c| match c {
                            AssistantContent::Text { text } => {
                                completion::AssistantContent::text(text)
                            }
                            AssistantContent::Refusal { refusal } => {
                                completion::AssistantContent::text(refusal)
                            }
                        })
                        .collect::<Vec<_>>();

                    content.extend(
                        tool_calls
                            .iter()
                            .map(|call| {
                                completion::AssistantContent::tool_call(
                                    &call.id,
                                    &call.function.name,
                                    call.function.arguments.clone(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    );
                    Ok(content)
                }
                _ => Err(CompletionError::ResponseError(
                    "Response did not contain a valid message or tool call".into(),
                )),
            }?;

            let choice = OneOrMany::many(content).map_err(|_| {
                CompletionError::ResponseError(
                    "Response contained no message or tool call (empty)".to_owned(),
                )
            })?;

            let usage = completion::Usage {
                input_tokens: response.usage.prompt_tokens as u64,
                output_tokens: response.usage.completion_tokens as u64,
                total_tokens: response.usage.total_tokens as u64,
            };

            Ok(completion::CompletionResponse {
                choice,
                usage,
                raw_response: response,
            })
        }
    }

    impl From<completion::ToolDefinition> for ToolDefinition {
        fn from(tool: completion::ToolDefinition) -> Self {
            Self {
                r#type: "function".into(),
                function: tool,
            }
        }
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    pub struct ToolDefinition {
        pub r#type: String,
        pub function: completion::ToolDefinition,
    }

    #[derive(Debug, Deserialize)]
    pub struct Function {
        pub name: String,
        pub arguments: String,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct CompletionResponse {
        pub id: String,
        pub model: String,
        pub choices: Vec<Choice>,
        pub created: i64,
        pub object: String,
        pub system_fingerprint: String,
        pub usage: Usage,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct Choice {
        pub finish_reason: String,
        pub index: i32,
        pub message: Message,
    }

    #[derive(Debug, Deserialize, Serialize)]
    pub struct Usage {
        pub completion_tokens: i32,
        pub prompt_tokens: i32,
        pub total_tokens: i32,
    }
}
