use std::collections::HashMap;

use crate::{
    message::{ToolCall, ToolFunction},
    providers::openai::{self},
    streaming,
};
use async_stream::stream;
use futures::StreamExt;
use reqwest::RequestBuilder;
use serde_json::Value;

use crate::{
    completion::{CompletionError, CompletionRequest},
    streaming::{StreamingCompletionModel, StreamingResult},
};
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

#[derive(Serialize, Deserialize, Debug)]
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

pub fn openaify_request(request: serde_json::Value) -> serde_json::Value {
    let mut request = request;
    let obj = request.as_object_mut().unwrap();

    // Transform messages array
    if let Some(messages) = obj.get_mut("messages") {
        if let Some(messages_array) = messages.as_array_mut() {
            for message in messages_array {
                if let Some(content) = message.get_mut("content") {
                    if let Some(content_array) = content.as_array() {
                        // If content is an array, extract text content
                        let text_content = content_array
                            .iter()
                            .filter_map(|item| {
                                if item.get("type")? == "text" {
                                    item.get("text").map(|t| t.as_str().unwrap_or_default())
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("");

                        if text_content.is_empty() {
                            *content = serde_json::Value::Null;
                        } else {
                            *content = serde_json::Value::String(text_content);
                        }
                    }
                }
            }
        }
    }

    request
}

impl StreamingCompletionModel for super::CompletionModel {
    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let filler = &openai::client::Client::from_env().completion_model(&self.model);
        let mut request = openai::completion::CompletionModel::create_completion_request(
            filler,
            completion_request,
        )?;

        if std::env::var("DEBUG").is_ok() {
            // debug
            let mut req_clone = request.clone();
            req_clone.as_object_mut().unwrap().remove("tools");
            // println!(
            //     "RIG request: {}",
            //     serde_json::to_string_pretty(&req_clone).unwrap()
            // );
            println!(
                "request: {}",
                serde_json::to_string_pretty(&openaify_request(req_clone)).unwrap()
            );
        }

        request
            .as_object_mut()
            .unwrap()
            .insert("stream".to_string(), serde_json::Value::Bool(true));

        let builder = self
            .client
            .post("/chat/completions")
            .json(&openaify_request(request));
        send_streaming_request(builder).await
    }
}

pub async fn send_streaming_request(
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
        let mut tool_calls = HashMap::new();
        let mut partial_line = String::new();

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
                    Err(e) => {
                        println!("RAW: {}", serde_json::to_string_pretty(
                        &serde_json::from_str::<serde_json::Value>(
                            &line
                        ).unwrap_or(
                            serde_json::json!({"line": line})
                        )).unwrap_or_default());
                        eprintln!("ERROR: {}", e);
                        continue;
                    }
                };

                if std::env::var("DEBUG").is_ok() {
                    println!("PARSED: {}", serde_json::to_string_pretty(&data).unwrap_or_default());
                }

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
                                function: ToolFunction {
                                    name: String::new(),
                                    arguments: serde_json::Value::Null,
                                },
                            });

                            // Update fields if present
                            if let Some(id) = &tool_call.id {
                                existing_tool_call.id = id.clone();
                            }
                            if let Some(name) = &tool_call.function.name {
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
                                let combined = format!("{}{}", current_args, chunk);

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

                    if let Some(content) = &delta.content {
                        if !content.is_empty() {
                            yield Ok(streaming::StreamingChoice::Message(content.clone()))
                        }
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
                                    Err(_) => serde_json::Value::String(args.clone()),
                                }
                            } else {
                                serde_json::Value::Null
                            };
                            let index = tool_call.index;

                            tool_calls.insert(index, ToolCall{
                                id: id.unwrap_or_default(),
                                function: ToolFunction {
                                    name: name.unwrap_or_default(),
                                    arguments,
                                },
                            });
                        }
                    }

                    if !message.content.is_empty() {
                        yield Ok(streaming::StreamingChoice::Message(message.content.clone()))
                    }
                }
            }
        }
        if std::env::var("DEBUG").is_ok() {
            println!("TOOL CALLS: {:?}", tool_calls);
        }
        if tool_calls.len() == 1 {
            let (_, tool_call) = tool_calls.into_iter().next().unwrap();
            yield Ok(streaming::StreamingChoice::ToolCall(tool_call.function.name, tool_call.id, tool_call.function.arguments));
        } else if !tool_calls.is_empty() {
            yield Ok(streaming::StreamingChoice::ParToolCall(tool_calls));
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openaify_request() {
        let faulty_request = serde_json::json!({
            "model": "google/gemini-2.0-flash-001",
          "messages": [
            {
              "role": "system",
              "content": [
                {
                  "type": "text",
                  "text": "you are a solana trading agent"
                }
              ]
            },
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": "\n                we are testing the resoning loop, fetch my solana balance three times\n                "
                }
              ]
            },
            {
              "role": "assistant",
              "content": [],
              "tool_calls": [
                {
                  "id": "tool_0_get_sol_balance",
                  "type": "function",
                  "function": {
                    "name": "get_sol_balance",
                    "arguments": "{}"
                  }
                }
              ]
            },
            {
              "role": "tool",
              "tool_call_id": "tool_0_get_sol_balance",
              "content": [
                {
                  "type": "text",
                  "text": "18391337"
                }
              ]
            },
          ],
          "tool_choice": "auto",
          "tools": [],
        });

        let want = serde_json::json!({
          "model": "google/gemini-2.0-flash-001",
          "messages": [
            {
              "role": "system",
              "content": "you are a solana trading agent"
            },
            {
                "role": "user",
                "content": "\n                we are testing the resoning loop, fetch my solana balance three times\n                "
            },
            {
              "role": "assistant",
              "content": null,
              "tool_calls": [
                {
                  "id": "tool_0_get_sol_balance",
                  "type": "function",
                  "function": {
                    "name": "get_sol_balance",
                    "arguments": "{}"
                  }
                }
              ]
            },
            {
              "role": "tool",
              "tool_call_id": "tool_0_get_sol_balance",
              "content": "18391337"
            }
          ],
          "tools": [],
          "tool_choice": "auto"
        });

        let got = openaify_request(faulty_request);
        pretty_assertions::assert_eq!(got, want);
    }

    #[test]
    fn test_openaify_request_tool_response() {
        let faulty_request = serde_json::json!({
          "messages": [
            {
              "role": "user",
              "content": [
                {
                  "type": "text",
                  "text": "we are testing parallel tool calls, check my solana balance and USDC balance (EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v), do this in one response please"
                }
              ]
            },
            {
              "role": "assistant",
              "content": [
                {
                  "id": "tool_0_get_sol_balance",
                  "function": {
                    "name": "get_sol_balance",
                    "arguments": {}
                  }
                },
                {
                  "id": "tool_1_get_spl_token_balance",
                  "function": {
                    "name": "get_spl_token_balance",
                    "arguments": {
                      "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                    }
                  }
                }
              ]
            },
            {
              "role": "user",
              "content": [
                {
                  "type": "toolresult",
                  "id": "tool_0_get_sol_balance",
                  "content": [
                    {
                      "Text": {
                        "text": "18391337"
                      }
                    }
                  ]
                },
                {
                  "type": "toolresult",
                  "id": "tool_1_get_spl_token_balance",
                  "content": [
                    {
                      "Text": {
                        "text": "[\"778427\",6,\"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v\"]"
                      }
                    }
                  ]
                }
              ]
            },
            {
              "role": "assistant",
              "content": [
                {
                  "text": "Your SOL balance is 0.018391337 SOL. Your USDC balance is 778427, with 6 decimals, meaning 0.778427 USDC.\n"
                }
              ]
            }
          ]
        });

        let want = serde_json::json!({
            "messages": [
                {
                    "role": "user",
                    "content": "we are testing parallel tool calls, check my solana balance and USDC balance (EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v), do this in one response please"
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "id": "tool_0_get_sol_balance",
                            "function": {
                                "name": "get_sol_balance",
                                "arguments": {}
                            }
                        },
                        {
                            "id": "tool_1_get_spl_token_balance",
                            "function": {
                                "name": "get_spl_token_balance",
                                "arguments": {
                                    "mint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                                }
                            }
                        }
                    ]
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_0_get_sol_balance",
                    "content": "18391337"
                },
                {
                    "role": "tool",
                    "tool_call_id": "tool_1_get_spl_token_balance",
                    "content":  "[\"778427\",6,\"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v\"]"
                },
                {
                    "role": "assistant",
                    "content":  "Your SOL balance is 0.018391337 SOL. Your USDC balance is 778427, with 6 decimals, meaning 0.778427 USDC.\n"
                }
            ]
        });
        let got = openaify_request(faulty_request);
        pretty_assertions::assert_eq!(got, want);
    }
}
