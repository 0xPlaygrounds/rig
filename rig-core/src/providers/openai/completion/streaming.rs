use crate::completion::{CompletionError, CompletionRequest};
use crate::json_utils;
use crate::json_utils::merge;
use crate::providers::openai::completion::{CompletionModel, Usage};
use crate::streaming;
use crate::streaming::RawStreamingChoice;
use async_stream::stream;
use futures::StreamExt;
use reqwest::RequestBuilder;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tracing::debug;

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
    pub id: Option<String>,
    pub function: StreamingFunction,
}

#[derive(Deserialize, Debug)]
struct StreamingDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default, deserialize_with = "json_utils::null_or_vec")]
    tool_calls: Vec<StreamingToolCall>,
}

#[derive(Deserialize, Debug)]
struct StreamingChoice {
    delta: StreamingDelta,
}

#[derive(Deserialize, Debug)]
struct StreamingCompletionChunk {
    choices: Vec<StreamingChoice>,
    usage: Option<Usage>,
}

#[derive(Clone)]
pub struct StreamingCompletionResponse {
    pub usage: Usage,
}

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let mut request = self.create_completion_request(completion_request)?;
        request = merge(
            request,
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let builder = self.client.post("/chat/completions").json(&request);
        send_compatible_streaming_request(builder).await
    }
}

pub async fn send_compatible_streaming_request(
    request_builder: RequestBuilder,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError> {
    let response = request_builder.send().await?;

    if !response.status().is_success() {
        return Err(CompletionError::ProviderError(format!(
            "{}: {}",
            response.status(),
            response.text().await?
        )));
    }

    // Handle OpenAI Compatible SSE chunks
    let inner = Box::pin(stream! {
        let mut stream = response.bytes_stream();

        let mut final_usage = Usage {
            prompt_tokens: 0,
            total_tokens: 0
        };

        let mut partial_data = None;
        let mut calls: HashMap<usize, (String, String, String)> = HashMap::new();

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
                    let Some(data) = line.strip_prefix("data:") else {
                        continue;
                    };

                    let data = data.trim_start();

                    // Partial data, split somewhere in the middle
                    if !line.ends_with("}") {
                        partial_data = Some(data.to_string());
                    } else {
                        line = data.to_string();
                    }
                }

                let data = serde_json::from_str::<StreamingCompletionChunk>(&line);

                let Ok(data) = data else {
                    let err = data.unwrap_err();
                    debug!("Couldn't serialize data as StreamingCompletionChunk: {:?}", err);
                    continue;
                };


                if let Some(choice) = data.choices.first() {

                    let delta = &choice.delta;

                    if !delta.tool_calls.is_empty() {
                        for tool_call in &delta.tool_calls {
                            let function = tool_call.function.clone();
                            // Start of tool call
                            // name: Some(String)
                            // arguments: None
                            if function.name.is_some() && function.arguments.is_empty() {
                                let id = tool_call.id.clone().unwrap_or("".to_string());

                                calls.insert(tool_call.index, (id, function.name.clone().unwrap(), "".to_string()));
                            }
                            // Part of tool call
                            // name: None or Empty String
                            // arguments: Some(String)
                            else if function.name.clone().is_none_or(|s| s.is_empty()) && !function.arguments.is_empty() {
                                let Some((id, name, arguments)) = calls.get(&tool_call.index) else {
                                    debug!("Partial tool call received but tool call was never started.");
                                    continue;
                                };

                                let new_arguments = &tool_call.function.arguments;
                                let arguments = format!("{arguments}{new_arguments}");

                                calls.insert(tool_call.index, (id.clone(), name.clone(), arguments));
                            }
                            // Entire tool call
                            else {
                                let id = tool_call.id.clone().unwrap_or("".to_string());
                                let name = function.name.expect("function name should be present for complete tool call");
                                let arguments = function.arguments;
                                let Ok(arguments) = serde_json::from_str(&arguments) else {
                                    debug!("Couldn't serialize '{}' as a json value", arguments);
                                    continue;
                                };

                                yield Ok(streaming::RawStreamingChoice::ToolCall {id, name, arguments, call_id: None })
                            }
                        }
                    }

                    if let Some(content) = &choice.delta.content {
                        yield Ok(streaming::RawStreamingChoice::Message(content.clone()))
                    }
                }


                if let Some(usage) = data.usage {
                    final_usage = usage.clone();
                }
            }
        }

        for (_, (id, name, arguments)) in calls {
            let Ok(arguments) = serde_json::from_str(&arguments) else {
                continue;
            };

            yield Ok(RawStreamingChoice::ToolCall {id, name, arguments, call_id: None });
        }

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: final_usage.clone()
        }))
    });

    Ok(streaming::StreamingCompletionResponse::stream(inner))
}
