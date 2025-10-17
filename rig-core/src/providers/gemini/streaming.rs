use crate::telemetry::SpanCombinator;
use async_stream::stream;
use futures::StreamExt;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use tracing::info_span;

use super::completion::{
    CompletionModel, create_request_body,
    gemini_api_types::{ContentCandidate, Part, PartKind},
};
use crate::{
    completion::{CompletionError, CompletionRequest, GetTokenUsage},
    streaming::{self},
};

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
#[serde(rename_all = "camelCase")]
pub struct PartialUsage {
    pub total_token_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_content_token_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates_token_count: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thoughts_token_count: Option<i32>,
    pub prompt_token_count: i32,
}

impl GetTokenUsage for PartialUsage {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();

        usage.input_tokens = self.prompt_token_count as u64;
        usage.output_tokens = (self.cached_content_token_count.unwrap_or_default()
            + self.candidates_token_count.unwrap_or_default()
            + self.thoughts_token_count.unwrap_or_default()) as u64;
        usage.total_tokens = usage.input_tokens + usage.output_tokens;

        Some(usage)
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StreamGenerateContentResponse {
    /// Candidate responses from the model.
    pub candidates: Vec<ContentCandidate>,
    pub model_version: Option<String>,
    pub usage_metadata: Option<PartialUsage>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamingCompletionResponse {
    pub usage_metadata: PartialUsage,
}

impl GetTokenUsage for StreamingCompletionResponse {
    fn token_usage(&self) -> Option<crate::completion::Usage> {
        let mut usage = crate::completion::Usage::new();
        usage.total_tokens = self.usage_metadata.total_token_count as u64;
        usage.output_tokens = self
            .usage_metadata
            .candidates_token_count
            .map(|x| x as u64)
            .unwrap_or(0);
        usage.input_tokens = self.usage_metadata.prompt_token_count as u64;
        Some(usage)
    }
}

impl CompletionModel<reqwest::Client> {
    pub(crate) async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError>
    {
        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "gcp.gemini",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = &completion_request.preamble,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.input.messages = tracing::field::Empty,
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };
        let request = create_request_body(completion_request)?;

        span.record_model_input(&request.contents);

        tracing::debug!(
            "Sending completion request to Gemini API {}",
            serde_json::to_string_pretty(&request)?
        );

        // Build the request with proper headers for SSE
        let mut event_source = self
            .client
            .post_sse(&format!(
                "/v1beta/models/{}:streamGenerateContent",
                self.model
            ))
            .json(&request)
            .eventsource()
            .expect("Cloning request must always succeed");

        let stream = stream! {
            let mut text_response = String::new();
            let mut model_outputs: Vec<Part> = Vec::new();
            while let Some(event_result) = event_source.next().await {
                match event_result {
                    Ok(Event::Open) => {
                        tracing::debug!("SSE connection opened");
                        continue;
                    }
                    Ok(Event::Message(message)) => {
                        // Skip heartbeat messages or empty data
                        if message.data.trim().is_empty() {
                            continue;
                        }

                        let data = match serde_json::from_str::<StreamGenerateContentResponse>(&message.data) {
                            Ok(d) => d,
                            Err(error) => {
                                tracing::error!(?error, message = message.data, "Failed to parse SSE message");
                                continue;
                            }
                        };

                        // Process the response data
                        let Some(choice) = data.candidates.first() else {
                            tracing::debug!("There is no content candidate");
                            continue;
                        };

                        for part in &choice.content.parts {
                            match part {
                                Part {
                                    part: PartKind::Text(text),
                                    thought: Some(true),
                                    ..
                                } => {
                                    yield Ok(streaming::RawStreamingChoice::Reasoning { reasoning: text.clone(), id: None, signature: None });
                                },
                                Part {
                                    part: PartKind::Text(text),
                                    ..
                                } => {
                                    text_response += text;
                                    yield Ok(streaming::RawStreamingChoice::Message(text.clone()));
                                },
                                Part {
                                    part: PartKind::FunctionCall(function_call),
                                    ..
                                } => {
                                    model_outputs.push(part.clone());
                                    yield Ok(streaming::RawStreamingChoice::ToolCall {
                                        name: function_call.name.clone(),
                                        id: function_call.name.clone(),
                                        arguments: function_call.args.clone(),
                                        call_id: None
                                    });
                                },
                                part => {
                                    tracing::warn!(?part, "Unsupported response type with streaming");
                                }
                            }
                        }

                        if choice.content.parts.is_empty() {
                            tracing::trace!(reason = ?choice.finish_reason, "There is no part in the streaming content");
                        }

                        // Check if this is the final response
                        if choice.finish_reason.is_some() {
                            if !text_response.is_empty() {
                                model_outputs.push(Part { thought: None, thought_signature: None, part: PartKind::Text(text_response), additional_params: None });
                            }
                            let span = tracing::Span::current();
                            span.record_model_output(&model_outputs);
                            span.record_token_usage(&data.usage_metadata);
                            yield Ok(streaming::RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
                                usage_metadata: data.usage_metadata.unwrap_or_default()
                            }));
                            break;
                        }
                    }
                    Err(reqwest_eventsource::Error::StreamEnded) => {
                        break;
                    }
                    Err(error) => {
                        tracing::error!(?error, "SSE error");
                        yield Err(CompletionError::ResponseError(error.to_string()));
                        break;
                    }
                }
            }

            // Ensure event source is closed when stream ends
            event_source.close();
        };

        Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
            stream,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_deserialize_stream_response_with_single_text_part() {
        let json_data = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Hello, world!"}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(response.candidates[0].content.parts.len(), 1);

        if let Part {
            part: PartKind::Text(text),
            ..
        } = &response.candidates[0].content.parts[0]
        {
            assert_eq!(text, "Hello, world!");
        } else {
            panic!("Expected text part");
        }
    }

    #[test]
    fn test_deserialize_stream_response_with_multiple_text_parts() {
        let json_data = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Hello, "},
                        {"text": "world!"},
                        {"text": " How are you?"}
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 8,
                "totalTokenCount": 18
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        assert_eq!(response.candidates.len(), 1);
        assert_eq!(response.candidates[0].content.parts.len(), 3);

        // Verify all three text parts are present
        for (i, expected_text) in ["Hello, ", "world!", " How are you?"].iter().enumerate() {
            if let Part {
                part: PartKind::Text(text),
                ..
            } = &response.candidates[0].content.parts[i]
            {
                assert_eq!(text, expected_text);
            } else {
                panic!("Expected text part at index {}", i);
            }
        }
    }

    #[test]
    fn test_deserialize_stream_response_with_multiple_tool_calls() {
        let json_data = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"city": "San Francisco"}
                            }
                        },
                        {
                            "functionCall": {
                                "name": "get_temperature",
                                "args": {"location": "New York"}
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 50,
                "candidatesTokenCount": 20,
                "totalTokenCount": 70
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        assert_eq!(response.candidates[0].content.parts.len(), 2);

        // Verify first tool call
        if let Part {
            part: PartKind::FunctionCall(call),
            ..
        } = &response.candidates[0].content.parts[0]
        {
            assert_eq!(call.name, "get_weather");
        } else {
            panic!("Expected function call at index 0");
        }

        // Verify second tool call
        if let Part {
            part: PartKind::FunctionCall(call),
            ..
        } = &response.candidates[0].content.parts[1]
        {
            assert_eq!(call.name, "get_temperature");
        } else {
            panic!("Expected function call at index 1");
        }
    }

    #[test]
    fn test_deserialize_stream_response_with_mixed_parts() {
        let json_data = json!({
            "candidates": [{
                "content": {
                    "parts": [
                        {
                            "text": "Let me think about this...",
                            "thought": true
                        },
                        {
                            "text": "Here's my response: "
                        },
                        {
                            "functionCall": {
                                "name": "search",
                                "args": {"query": "rust async"}
                            }
                        },
                        {
                            "text": "I found the answer!"
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 100,
                "candidatesTokenCount": 50,
                "thoughtsTokenCount": 15,
                "totalTokenCount": 165
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        let parts = &response.candidates[0].content.parts;
        assert_eq!(parts.len(), 4);

        // Verify reasoning (thought) part
        if let Part {
            part: PartKind::Text(text),
            thought: Some(true),
            ..
        } = &parts[0]
        {
            assert_eq!(text, "Let me think about this...");
        } else {
            panic!("Expected thought part at index 0");
        }

        // Verify regular text
        if let Part {
            part: PartKind::Text(text),
            thought,
            ..
        } = &parts[1]
        {
            assert_eq!(text, "Here's my response: ");
            assert!(thought.is_none() || thought == &Some(false));
        } else {
            panic!("Expected text part at index 1");
        }

        // Verify tool call
        if let Part {
            part: PartKind::FunctionCall(call),
            ..
        } = &parts[2]
        {
            assert_eq!(call.name, "search");
        } else {
            panic!("Expected function call at index 2");
        }

        // Verify final text
        if let Part {
            part: PartKind::Text(text),
            ..
        } = &parts[3]
        {
            assert_eq!(text, "I found the answer!");
        } else {
            panic!("Expected text part at index 3");
        }
    }

    #[test]
    fn test_deserialize_stream_response_with_empty_parts() {
        let json_data = json!({
            "candidates": [{
                "content": {
                    "parts": [],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 0,
                "totalTokenCount": 10
            }
        });

        let response: StreamGenerateContentResponse = serde_json::from_value(json_data).unwrap();
        assert_eq!(response.candidates[0].content.parts.len(), 0);
    }

    #[test]
    fn test_partial_usage_token_calculation() {
        let usage = PartialUsage {
            total_token_count: 100,
            cached_content_token_count: Some(20),
            candidates_token_count: Some(30),
            thoughts_token_count: Some(10),
            prompt_token_count: 40,
        };

        let token_usage = usage.token_usage().unwrap();
        assert_eq!(token_usage.input_tokens, 40);
        assert_eq!(token_usage.output_tokens, 60); // 20 + 30 + 10
        assert_eq!(token_usage.total_tokens, 100);
    }

    #[test]
    fn test_partial_usage_with_missing_counts() {
        let usage = PartialUsage {
            total_token_count: 50,
            cached_content_token_count: None,
            candidates_token_count: Some(30),
            thoughts_token_count: None,
            prompt_token_count: 20,
        };

        let token_usage = usage.token_usage().unwrap();
        assert_eq!(token_usage.input_tokens, 20);
        assert_eq!(token_usage.output_tokens, 30); // Only candidates_token_count
        assert_eq!(token_usage.total_tokens, 50);
    }

    #[test]
    fn test_streaming_completion_response_token_usage() {
        let response = StreamingCompletionResponse {
            usage_metadata: PartialUsage {
                total_token_count: 150,
                cached_content_token_count: None,
                candidates_token_count: Some(75),
                thoughts_token_count: None,
                prompt_token_count: 75,
            },
        };

        let token_usage = response.token_usage().unwrap();
        assert_eq!(token_usage.input_tokens, 75);
        assert_eq!(token_usage.output_tokens, 75);
        assert_eq!(token_usage.total_tokens, 150);
    }
}
