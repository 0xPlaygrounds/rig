use super::completion::CompletionModel;
use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils;
use crate::json_utils::merge;
use crate::streaming;
use crate::streaming::{StreamingCompletionModel, StreamingResult};
use async_stream::stream;
use futures::StreamExt;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

// ================================================================
// OpenAI Completion Streaming API
// ================================================================
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct StreamingToolCall {
    pub index: usize,
    pub function: StreamingFunction,
}

#[derive(Deserialize)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize)]
struct StreamingCompletionResponse {
    choices: Vec<StreamingChoice>,
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut request = self.create_completion_request(completion_request)?;
        request = merge(request, json!({"stream": true}));

        let builder = self.client.post("/chat/completions").json(&request);
        send_compatible_streaming_request(builder).await
    }
}

pub async fn send_compatible_streaming_request(
    request_builder: RequestBuilder,
) -> Result<StreamingResult, CompletionError> {
    let response = request_builder.send().await?;

    if !response.status().is_success() {
        return Err(CompletionError::ProviderError(format!(
            "{}: {}",
            response.status(),
            response.text().await?
        )));
    }

    // Handle OpenAI Compatible SSE chunks
    Ok(Box::pin(stream! {
        let mut stream = response.bytes_stream();

        let mut partial_data = None;
        let mut calls: HashMap<usize, (String, String)> = HashMap::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    yield Err(CompletionError::from(e));
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



                // If there was a remaining part, concat with current line
                if partial_data.is_some() {
                    line = format!("{}{}", partial_data.unwrap(), line);
                    partial_data = None;
                }
                // Otherwise full data line
                else {
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };

                    // Partial data, split somewhere in the middle
                    if !line.ends_with("}") {
                        partial_data = Some(data.to_string());
                    } else {
                        line = data.to_string();
                    }
                }

                let data = serde_json::from_str::<StreamingCompletionResponse>(&line);

                let Ok(data) = data else {
                    continue;
                };

                let choice = data.choices.first().expect("Should have at least one choice");

                let delta = &choice.delta;

                if !delta.tool_calls.is_empty() {
                    for tool_call in &delta.tool_calls {
                        let function = tool_call.function.clone();

                        // Start of tool call
                        // name: Some(String)
                        // arguments: None
                        if function.name.is_some() && function.arguments.is_empty() {
                            calls.insert(tool_call.index, (function.name.clone().unwrap(), "".to_string()));
                        }
                        // Part of tool call
                        // name: None
                        // arguments: Some(String)
                        else if function.name.is_none() && !function.arguments.is_empty() {
                            let Some((name, arguments)) = calls.get(&tool_call.index) else {
                                continue;
                            };

                            let new_arguments = &tool_call.function.arguments;
                            let arguments = format!("{}{}", arguments, new_arguments);

                            calls.insert(tool_call.index, (name.clone(), arguments));
                        }
                        // Entire tool call
                        else {
                            let name = function.name.unwrap();
                            let arguments = function.arguments;
                            let Ok(arguments) = serde_json::from_str(&arguments) else {
                                continue;
                            };

                            yield Ok(streaming::StreamingChoice::ToolCall(name, "".to_string(), arguments))
                        }
                    }
                }

                if let Some(content) = &choice.delta.content {
                    yield Ok(streaming::StreamingChoice::Message(content.clone()))
                }
            }
        }

        for (_, (name, arguments)) in calls {
            let Ok(arguments) = serde_json::from_str(&arguments) else {
                continue;
            };

            yield Ok(streaming::StreamingChoice::ToolCall(name, "".to_string(), arguments))
        }
    }))
}
