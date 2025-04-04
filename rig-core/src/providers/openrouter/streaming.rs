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
    pub message: MessageResponse,
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
pub struct OpenRouterToolCall {
    pub index: usize,
    pub id: String,
    pub r#type: String,
    pub function: ToolFunction,
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
        let request = openai::completion::CompletionModel::create_completion_request(
            filler,
            completion_request,
        )?;

        if std::env::var("DEBUG").is_ok() {
            // debug
            let mut req_clone = request.clone();
            req_clone.as_object_mut().unwrap().remove("tools");
            println!(
                "RIG request: {}",
                serde_json::to_string_pretty(&req_clone).unwrap()
            );
            println!(
                "OPENAIFIED request: {}",
                serde_json::to_string_pretty(&openaify_request(req_clone)).unwrap()
            );
        }

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

        let mut partial_data = None;
        let mut tool_calls = HashMap::new();

        while let Some(chunk_result) = stream.next().await {
            println!("chunk_result: {:?}", chunk_result);
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
                    let data = line.replace("data: ", "");

                    // Partial data, split somewhere in the middle
                    if !line.ends_with("}") {
                        partial_data = Some(data.to_string());
                    } else {
                        line = data.to_string();
                    }
                }

                if std::env::var("DEBUG").is_ok() {
                    println!("RAW: {}", serde_json::to_string_pretty(
                        &serde_json::from_str::<serde_json::Value>(
                            &line
                        ).unwrap_or(
                            serde_json::json!({"line": line}
                        ))).unwrap_or_default());
                }

                if line.trim() == "" {
                    continue;
                }

                let data = match serde_json::from_str::<StreamingCompletionResponse>(&line) {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("ERROR: {}", e);
                        continue;
                    }
                };

                if std::env::var("DEBUG").is_ok() {
                    println!("PARSED: {}", serde_json::to_string_pretty(&data).unwrap_or_default());
                }

                let choice = data.choices.first().expect("Should have at least one choice");

                if !choice.message.tool_calls.is_empty() {
                    for tool_call in &choice.message.tool_calls {
                        let name = tool_call.function.name.clone();
                        let id = tool_call.id.clone();
                        let arguments = match &tool_call.function.arguments {
                            serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(&s) {
                                Ok(v) => v,
                                Err(e) => {
                                    eprintln!("Failed to parse tool arguments: {}", e);
                                    continue;
                                }
                            },
                            v => v.clone(),
                        };
                        let index = tool_call.index;

                        tool_calls.insert(index, ToolCall{
                            id,
                            function: ToolFunction {
                                name: name.clone(),
                                arguments,
                            },
                        });
                    }
                }

                if !choice.message.content.is_empty() {
                    yield Ok(streaming::StreamingChoice::Message(choice.message.content.clone()))
                }
            }
        }
        if std::env::var("DEBUG").is_ok() {
            println!("TOOL CALLS: {:?}", tool_calls);
        }
        if tool_calls.len() == 1 {
            let (index, tool_call) = tool_calls.into_iter().next().unwrap();
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
}
