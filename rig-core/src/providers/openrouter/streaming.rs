use http::Request;
use std::collections::HashMap;
use tracing::info_span;

use crate::{
    completion::GetTokenUsage,
    http_client::{self, HttpClientExt},
    json_utils,
    message::{ToolCall, ToolFunction},
    streaming::{self},
};
use async_stream::stream;
use futures::StreamExt;
use serde_json::{Value, json};

use crate::completion::{CompletionError, CompletionRequest};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct StreamingCompletionResponse {
    pub id: String,
    pub choices: Vec<StreamingChoice>,
    pub created: u64,
    pub model: String,
    pub object: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ResponseUsage>,
}

impl GetTokenUsage for FinalCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.usage.prompt_tokens as u64;
        usage.output_tokens = self.usage.completion_tokens as u64;
        usage.total_tokens = self.usage.total_tokens as u64;

        Some(usage)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct StreamingChoice {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<MessageResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<DeltaResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorResponse>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MessageResponse {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<Value>,
    #[serde(default)]
    pub tool_calls: Vec<OpenRouterToolCall>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenRouterToolFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OpenRouterToolCall {
    pub index: usize,
    pub id: Option<String>,
    pub r#type: Option<String>,
    pub function: OpenRouterToolFunction,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct ResponseUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ErrorResponse {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, Value>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DeltaResponse {
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OpenRouterToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_finish_reason: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct FinalCompletionResponse {
    pub usage: ResponseUsage,
}

impl<T> super::CompletionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + 'static,
{
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<FinalCompletionResponse>, CompletionError>
    {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;

        let request = json_utils::merge(request, json!({"stream": true}));

        let body = serde_json::to_vec(&request)?;

        let req = self
            .client
            .post("/chat/completions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|x| CompletionError::HttpError(x.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "openrouter",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(request.get("messages").unwrap()).unwrap(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        tracing::Instrument::instrument(
            send_streaming_request(self.client.http_client.clone(), req),
            span,
        )
        .await
    }
}

pub async fn send_streaming_request<T>(
    client: T,
    req: Request<Vec<u8>>,
) -> Result<streaming::StreamingCompletionResponse<FinalCompletionResponse>, CompletionError>
where
    T: HttpClientExt + Clone + 'static,
{
    let response = client.send_streaming(req).await?;
    let status = response.status();

    if !status.is_success() {
        return Err(CompletionError::ProviderError(format!(
            "Got response error trying to send a completion request to OpenRouter: {status}"
        )));
    }

    let mut stream = response.into_body();

    // Handle OpenAI Compatible SSE chunks
    let stream = stream! {
        let mut tool_calls = HashMap::new();
        let mut partial_line = String::new();
        let mut final_usage = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(CompletionError::from(http_client::Error::Instance(e.into())));
                    break;
                }
            };

            let text = match String::from_utf8(chunk.to_vec()) {
                Ok(t) => t,
                Err(e) => {
                    yield Err(CompletionError::ResponseError(e.to_string()));
                    break;
                }
            };

            for line in text.lines() {
                let mut line = line.to_string();

                // Skip empty lines and processing messages, as well as [DONE] (might be useful though)
                if line.trim().is_empty() || line.trim() == ": OPENROUTER PROCESSING" || line.trim() == "data: [DONE]" {
                    continue;
                }

                // Handle data: prefix
                line = line.strip_prefix("data: ").unwrap_or(&line).to_string();

                // If line starts with { but doesn't end with }, it's a partial JSON
                if line.starts_with('{') && !line.ends_with('}') {
                    partial_line = line;
                    continue;
                }

                // If we have a partial line and this line ends with }, complete it
                if !partial_line.is_empty() {
                    if line.ends_with('}') {
                        partial_line.push_str(&line);
                        line = partial_line;
                        partial_line = String::new();
                    } else {
                        partial_line.push_str(&line);
                        continue;
                    }
                }

                let data = match serde_json::from_str::<StreamingCompletionResponse>(&line) {
                    Ok(data) => data,
                    Err(_) => {
                        continue;
                    }
                };


                let choice = data.choices.first().expect("Should have at least one choice");

                // TODO this has to handle outputs like this:
                // [{"index": 0, "id": "call_DdmO9pD3xa9XTPNJ32zg2hcA", "function": {"arguments": "", "name": "get_weather"}, "type": "function"}]
                // [{"index": 0, "id": null, "function": {"arguments": "{\"", "name": null}, "type": null}]
                // [{"index": 0, "id": null, "function": {"arguments": "location", "name": null}, "type": null}]
                // [{"index": 0, "id": null, "function": {"arguments": "\":\"", "name": null}, "type": null}]
                // [{"index": 0, "id": null, "function": {"arguments": "Paris", "name": null}, "type": null}]
                // [{"index": 0, "id": null, "function": {"arguments": ",", "name": null}, "type": null}]
                // [{"index": 0, "id": null, "function": {"arguments": " France", "name": null}, "type": null}]
                // [{"index": 0, "id": null, "function": {"arguments": "\"}", "name": null}, "type": null}]
                if let Some(delta) = &choice.delta {
                    if !delta.tool_calls.is_empty() {
                        for tool_call in &delta.tool_calls {
                            let index = tool_call.index;

                            // Get or create tool call entry
                            let existing_tool_call = tool_calls.entry(index).or_insert_with(|| ToolCall {
                                id: String::new(),
                                call_id: None,
                                function: ToolFunction {
                                    name: String::new(),
                                    arguments: serde_json::Value::Null,
                                },
                            });

                            // Update fields if present
                            if let Some(id) = &tool_call.id && !id.is_empty() {
                                    existing_tool_call.id = id.clone();
                            }

                            if let Some(name) = &tool_call.function.name && !name.is_empty() {
                                    existing_tool_call.function.name = name.clone();
                            }

                            if let Some(chunk) = &tool_call.function.arguments {
                                // Convert current arguments to string if needed
                                let current_args = match &existing_tool_call.function.arguments {
                                    serde_json::Value::Null => String::new(),
                                    serde_json::Value::String(s) => s.clone(),
                                    v => v.to_string(),
                                };

                                // Concatenate the new chunk
                                let combined = format!("{current_args}{chunk}");

                                // Try to parse as JSON if it looks complete
                                if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
                                    match serde_json::from_str(&combined) {
                                        Ok(parsed) => existing_tool_call.function.arguments = parsed,
                                        Err(_) => existing_tool_call.function.arguments = serde_json::Value::String(combined),
                                    }
                                } else {
                                    existing_tool_call.function.arguments = serde_json::Value::String(combined);
                                }
                            }
                        }
                    }

                    if let Some(content) = &delta.content &&!content.is_empty() {
                            yield Ok(streaming::RawStreamingChoice::Message(content.clone()))
                    }

                    if let Some(usage) = data.usage {
                        final_usage = Some(usage);
                    }
                }

                // Handle message format
                if let Some(message) = &choice.message {
                    if !message.tool_calls.is_empty() {
                        for tool_call in &message.tool_calls {
                            let name = tool_call.function.name.clone();
                            let id = tool_call.id.clone();
                            let arguments = if let Some(args) = &tool_call.function.arguments {
                                // Try to parse the string as JSON, fallback to string value
                                match serde_json::from_str(args) {
                                    Ok(v) => v,
                                    Err(_) => serde_json::Value::String(args.to_string()),
                                }
                            } else {
                                serde_json::Value::Null
                            };
                            let index = tool_call.index;

                            tool_calls.insert(index, ToolCall {
                                id: id.unwrap_or_default(),
                                call_id: None,
                                function: ToolFunction {
                                    name: name.unwrap_or_default(),
                                    arguments,
                                },
                            });
                        }
                    }

                    if !message.content.is_empty() {
                        yield Ok(streaming::RawStreamingChoice::Message(message.content.clone()))
                    }
                }
            }
        }

        for (_, tool_call) in tool_calls.into_iter() {

            yield Ok(streaming::RawStreamingChoice::ToolCall{
                name: tool_call.function.name,
                id: tool_call.id,
                arguments: tool_call.function.arguments,
                call_id: None
            });
        }

        yield Ok(streaming::RawStreamingChoice::FinalResponse(FinalCompletionResponse {
            usage: final_usage.unwrap_or_default()
        }))

    };

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}
