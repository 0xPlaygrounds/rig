use reqwest_eventsource::{Event, RequestBuilderExt};
use std::collections::HashMap;
use tracing::info_span;

use crate::{
    completion::GetTokenUsage,
    http_client, json_utils,
    message::{ToolCall, ToolFunction},
    streaming::{self},
};
use async_stream::stream;
use futures::StreamExt;
use reqwest::RequestBuilder;
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

impl super::CompletionModel<reqwest::Client> {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<FinalCompletionResponse>, CompletionError>
    {
        let preamble = completion_request.preamble.clone();
        let request = self.create_completion_request(completion_request)?;

        let request = json_utils::merge(request, json!({"stream": true}));

        let builder = self
            .client
            .reqwest_post("/chat/completions")
            .header("Content-Type", "application/json")
            .json(&request);

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

        tracing::Instrument::instrument(send_streaming_request(builder), span).await
    }
}

pub async fn send_streaming_request(
    request_builder: RequestBuilder,
) -> Result<streaming::StreamingCompletionResponse<FinalCompletionResponse>, CompletionError> {
    let response = request_builder
        .send()
        .await
        .map_err(|e| CompletionError::HttpError(http_client::Error::Instance(e.into())))?;

    if !response.status().is_success() {
        return Err(CompletionError::ProviderError(format!(
            "{}: {}",
            response.status(),
            response
                .text()
                .await
                .map_err(|e| CompletionError::HttpError(http_client::Error::Instance(e.into())))?
        )));
    }

    // Handle OpenAI Compatible SSE chunks
    let stream = stream! {
        let mut stream = response.bytes_stream();
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

pub async fn send_streaming_request1(
    request_builder: RequestBuilder,
) -> Result<streaming::StreamingCompletionResponse<FinalCompletionResponse>, CompletionError> {
    let mut event_source = request_builder
        .eventsource()
        .expect("Cloning request must always succeed");

    let stream = stream! {
        // Accumulate tool calls by index while streaming
        let mut tool_calls: HashMap<usize, ToolCall> = HashMap::new();
        let mut final_usage = None;

        while let Some(event_result) = event_source.next().await {
            match event_result {
                Ok(Event::Open) => {
                    tracing::trace!("SSE connection opened");
                    continue;
                }

                Ok(Event::Message(event_message)) => {
                    let raw = event_message.data;

                    let parsed = serde_json::from_str::<StreamingCompletionResponse>(&raw);
                    let Ok(data) = parsed else {
                        tracing::debug!("Couldn't parse OpenRouter payload as StreamingCompletionResponse; skipping chunk");
                        continue;
                    };

                    // Expect at least one choice (keeps original behavior)
                    let choice = match data.choices.first() {
                        Some(c) => c,
                        None => continue,
                    };

                    // --- Handle delta (streaming updates) ---
                    if let Some(delta) = &choice.delta {
                        if !delta.tool_calls.is_empty() {
                            for tc in &delta.tool_calls {
                                let index = tc.index;

                                // Ensure entry exists
                                let existing = tool_calls.entry(index).or_insert_with(|| ToolCall {
                                    id: String::new(),
                                    call_id: None,
                                    function: ToolFunction {
                                        name: String::new(),
                                        arguments: Value::Null,
                                    },
                                });

                                // Update id if present and non-empty
                                if let Some(id) = &tc.id && !id.is_empty() {
                                        existing.id = id.clone();
                                }

                                // Update name if present and non-empty
                                if let Some(name) = &tc.function.name && !name.is_empty() {
                                    existing.function.name = name.clone();
                                }

                                // Append argument chunk if present
                                if let Some(chunk) = &tc.function.arguments {
                                    // Current arguments as string (or empty)
                                    let current_args = match &existing.function.arguments {
                                        Value::Null => String::new(),
                                        Value::String(s) => s.clone(),
                                        v => v.to_string(),
                                    };

                                    let combined = format!("{}{}", current_args, chunk);

                                    // If it looks like complete JSON object, try parse
                                    if combined.trim_start().starts_with('{') && combined.trim_end().ends_with('}') {
                                        match serde_json::from_str::<Value>(&combined) {
                                            Ok(parsed_value) => existing.function.arguments = parsed_value,
                                            Err(_) => existing.function.arguments = Value::String(combined),
                                        }
                                    } else {
                                        existing.function.arguments = Value::String(combined);
                                    }
                                }
                            }
                        }

                        // Streamed text content
                        if let Some(content) = &delta.content && !content.is_empty() {
                            yield Ok(streaming::RawStreamingChoice::Message(content.clone()));
                        }

                        // usage update (if present)
                        if let Some(usage) = data.usage {
                            final_usage = Some(usage);
                        }
                    }

                    // --- Handle message (final/other message structure) ---
                    if let Some(message) = &choice.message {
                        if !message.tool_calls.is_empty() {
                            for tc in &message.tool_calls {
                                let idx = tc.index;
                                let name = tc.function.name.clone().unwrap_or_default();
                                let id = tc.id.clone().unwrap_or_default();

                                let args_value = if let Some(args_str) = &tc.function.arguments {
                                    match serde_json::from_str::<Value>(args_str) {
                                        Ok(v) => v,
                                        Err(_) => Value::String(args_str.clone()),
                                    }
                                } else {
                                    Value::Null
                                };

                                tool_calls.insert(idx, ToolCall {
                                    id,
                                    call_id: None,
                                    function: ToolFunction {
                                        name,
                                        arguments: args_value,
                                    },
                                });
                            }
                        }

                        if !message.content.is_empty() {
                            yield Ok(streaming::RawStreamingChoice::Message(message.content.clone()));
                        }
                    }
                }

                Err(reqwest_eventsource::Error::StreamEnded) => {
                    break;
                }

                Err(error) => {
                    tracing::error!(?error, "SSE error from OpenRouter event source");
                    yield Err(CompletionError::ResponseError(error.to_string()));
                    break;
                }
            }
        }

        // Ensure event source is closed when stream ends
        event_source.close();

        // Flush any accumulated tool calls (that weren't emitted as ToolCall earlier)
        for (_idx, tool_call) in tool_calls.into_iter() {
            yield Ok(streaming::RawStreamingChoice::ToolCall {
                name: tool_call.function.name,
                id: tool_call.id,
                arguments: tool_call.function.arguments,
                call_id: None,
            });
        }

        // Final response with usage
        yield Ok(streaming::RawStreamingChoice::FinalResponse(FinalCompletionResponse {
            usage: final_usage.unwrap_or_default(),
        }));
    };

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream,
    )))
}
