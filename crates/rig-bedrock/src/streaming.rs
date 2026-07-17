use crate::types::completion_request::AwsCompletionRequest;
use crate::{
    completion::{CompletionModel, resolve_request_model},
    types::errors::{AwsSdkConverseStreamError, converse_stream_output_completion_error},
};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::completion::{CompletionTerminalMetadata, GetCompletionMetadata};
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use rig_core::{
    completion::CompletionError,
    message::ReasoningContent,
    streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent},
};
use serde::{Deserialize, Serialize};
use tracing_futures::Instrument;

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<BedrockUsage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub terminal_metadata: Option<CompletionTerminalMetadata>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_write_input_tokens: Option<i32>,
}

impl GetCompletionMetadata for BedrockStreamingResponse {
    fn token_usage(&self) -> rig_core::completion::Usage {
        self.usage
            .as_ref()
            .map(|u| rig_core::completion::Usage {
                input_tokens: u.input_tokens as u64,
                output_tokens: u.output_tokens as u64,
                total_tokens: u.total_tokens as u64,
                cached_input_tokens: u.cache_read_input_tokens.unwrap_or_default() as u64,
                cache_creation_input_tokens: u.cache_write_input_tokens.unwrap_or_default() as u64,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            })
            .unwrap_or_default()
    }

    fn terminal_metadata(&self) -> Option<CompletionTerminalMetadata> {
        self.terminal_metadata.clone()
    }
}

fn terminal_metadata_from_stop_reason(
    reason: &aws_bedrock::StopReason,
) -> CompletionTerminalMetadata {
    crate::terminal_metadata::from_stop_reason(reason.as_str())
}

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    internal_call_id: String,
    input_json: String,
}

#[derive(Default)]
struct ReasoningState {
    content: String,
    signature: Option<String>,
}

/// Convert an accumulated [`ReasoningState`] into a streaming reasoning chunk.
///
/// Adaptive-thinking blocks from Bedrock can arrive as signature-only — i.e. a
/// `Signature` delta with no preceding non-empty `Text` delta. Dropping such
/// blocks loses the signature on the way back to the consumer, which then
/// fails on the next turn with `messages.N.content.0.thinking.signature:
/// Field required` when the conversation is replayed to Bedrock. We must emit
/// whenever either the content or the signature is present; both-empty is
/// still skipped.
fn finalize_reasoning(
    state: ReasoningState,
) -> Option<RawStreamingChoice<BedrockStreamingResponse>> {
    if state.content.is_empty() && state.signature.is_none() {
        return None;
    }
    Some(RawStreamingChoice::Reasoning {
        id: None,
        content: ReasoningContent::Text {
            text: state.content,
            signature: state.signature,
        },
    })
}

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: rig_core::completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<BedrockStreamingResponse>, CompletionError> {
        let request_model = resolve_request_model(&self.model, &completion_request);
        let system_instructions = completion_request.preamble.clone();
        let record_telemetry_content = completion_request.record_telemetry_content;
        let request = AwsCompletionRequest {
            inner: completion_request,
            prompt_caching: self.prompt_caching,
        };
        let span = CompletionSpanBuilder::new(
            "aws_bedrock",
            &request_model,
            CompletionOperation::ChatStreaming,
        )
        .system_instructions(system_instructions.as_deref(), record_telemetry_content)
        .build();

        let mut converse_builder = self
            .client
            .get_inner()
            .await
            .converse_stream()
            .model_id(request_model);

        let tool_config = request.tools_config()?;
        let prompt_with_history = request.messages()?;
        let output_config = request.output_config()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt()?)
            .set_messages(Some(prompt_with_history))
            .set_output_config(output_config);

        let response = converse_builder
            .send()
            .instrument(span.clone())
            .await
            .map_err(|sdk_error| {
                Into::<CompletionError>::into(AwsSdkConverseStreamError(sdk_error))
            })?;

        let stream = Box::pin(stream! {
            let span = tracing::Span::current();
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut current_reasoning: Option<ReasoningState> = None;
            let mut terminal_metadata: Option<CompletionTerminalMetadata> = None;
            let mut stream = response.stream;
            loop {
                let output = match stream.recv().await {
                    Ok(Some(output)) => output,
                    Ok(None) => break,
                    Err(err) => {
                        yield Err(converse_stream_output_completion_error(err.into_service_error()));
                        break;
                    }
                };
                match output {
                    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
                        let delta = event.delta.ok_or(CompletionError::ProviderError("The delta for a content block is missing".into()))?;
                        match delta {
                            aws_bedrock::ContentBlockDelta::Text(text) => {
                                if current_tool_call.is_none() {
                                    yield Ok(RawStreamingChoice::Message(text))
                                }
                            },
                            aws_bedrock::ContentBlockDelta::ToolUse(tool) => {
                                if let Some(ref mut tool_call) = current_tool_call {
                                    let delta = tool.input().to_string();
                                    tool_call.input_json.push_str(&delta);

                                    // Emit the delta so UI can show progress
                                    yield Ok(RawStreamingChoice::ToolCallDelta {
                                        id: tool_call.id.clone(),
                                        internal_call_id: tool_call.internal_call_id.clone(),
                                        content: ToolCallDeltaContent::Delta(delta),
                                    });
                                }
                            },
                            aws_bedrock::ContentBlockDelta::ReasoningContent(reasoning) => {
                                match reasoning {
                                    aws_bedrock::ReasoningContentBlockDelta::Text(text) => {
                                        if current_reasoning.is_none() {
                                            current_reasoning = Some(ReasoningState::default());
                                        }

                                        if let Some(ref mut state) = current_reasoning {
                                            state.content.push_str(text.as_str());
                                        }

                                        if !text.is_empty() {
                                            yield Ok(RawStreamingChoice::ReasoningDelta {
                                                reasoning: text.clone(),
                                                id: None,
                                            })
                                        }
                                    },
                                    aws_bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                                        if current_reasoning.is_none() {
                                            current_reasoning = Some(ReasoningState::default());
                                        }

                                        if let Some(ref mut state) = current_reasoning {
                                            state.signature = Some(signature.clone());
                                        }
                                    },
                                    _ => {}
                                }
                            },
                            _ => {}
                        }
                    },
                    aws_bedrock::ConverseStreamOutput::ContentBlockStart(event) => {
                        match event.start.ok_or(CompletionError::ProviderError("ContentBlockStart has no data".into()))? {
                            aws_bedrock::ContentBlockStart::ToolUse(tool_use) => {
                                let internal_call_id = rig_core::id::generate();
                                current_tool_call = Some(ToolCallState {
                                    name: tool_use.name.clone(),
                                    id: tool_use.tool_use_id.clone(),
                                    internal_call_id: internal_call_id.clone(),
                                    input_json: String::new(),
                                });
                                yield Ok(RawStreamingChoice::ToolCallDelta {
                                    id: tool_use.tool_use_id,
                                    internal_call_id,
                                    content: ToolCallDeltaContent::Name(tool_use.name),
                                });
                            },
                            _ => yield Err(CompletionError::ProviderError("Stream is empty".into()))
                        }
                    },
                    aws_bedrock::ConverseStreamOutput::ContentBlockStop(_event) => {
                        if let Some(reasoning_state) = current_reasoning.take()
                            && let Some(choice) = finalize_reasoning(reasoning_state) {
                                yield Ok(choice)
                            }
                    },
                    aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => {
                        terminal_metadata = Some(terminal_metadata_from_stop_reason(
                            &message_stop_event.stop_reason,
                        ));
                        match message_stop_event.stop_reason {
                            aws_bedrock::StopReason::ToolUse => {
                                if let Some(tool_call) = current_tool_call.take() {
                                    // Handle empty input_json for tools with no parameters
                                    let tool_input = if tool_call.input_json.is_empty() {
                                        serde_json::json!({})
                                    } else {
                                        serde_json::from_str(tool_call.input_json.as_str())?
                                    };
                                    yield Ok(RawStreamingChoice::ToolCall(
                                        RawStreamingToolCall::new(tool_call.id, tool_call.name, tool_input)
                                            .with_internal_call_id(tool_call.internal_call_id)
                                    ));
                                } else {
                                    yield Err(CompletionError::ProviderError("Failed to call tool".into()))
                                }
                            }
                            aws_bedrock::StopReason::MalformedModelOutput
                            | aws_bedrock::StopReason::MalformedToolUse => {
                                yield Err(CompletionError::ProviderError(format!(
                                    "Bedrock stopped with {}",
                                    message_stop_event.stop_reason.as_str()
                                )));
                                return;
                            }
                            _ => {}
                        }
                    },
                    aws_bedrock::ConverseStreamOutput::Metadata(metadata_event) => {
                        // Extract usage information from metadata
                        let final_response = BedrockStreamingResponse {
                            usage: metadata_event.usage.map(|usage| BedrockUsage {
                                    input_tokens: usage.input_tokens,
                                    output_tokens: usage.output_tokens,
                                    total_tokens: usage.total_tokens,
                                    cache_read_input_tokens: usage.cache_read_input_tokens,
                                    cache_write_input_tokens: usage.cache_write_input_tokens,
                                }),
                            terminal_metadata: terminal_metadata.clone(),
                        };
                        span.record_token_usage(&final_response);
                        yield Ok(RawStreamingChoice::FinalResponse(final_response));
                    },
                    _ => {}
                }
            }
        }.instrument(span));

        Ok(StreamingCompletionResponse::stream(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rig_core::completion::CompletionFinishReason;

    #[test]
    fn test_bedrock_usage_creation() {
        let usage = BedrockUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cache_read_input_tokens: None,
            cache_write_input_tokens: None,
        };

        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn test_bedrock_streaming_response_with_usage() {
        let response = BedrockStreamingResponse {
            usage: Some(BedrockUsage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
                cache_read_input_tokens: Some(40),
                cache_write_input_tokens: Some(10),
            }),
            terminal_metadata: None,
        };

        assert_eq!(
            response.token_usage(),
            rig_core::completion::Usage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
                cached_input_tokens: 40,
                cache_creation_input_tokens: 10,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
    }

    #[test]
    fn test_bedrock_streaming_response_without_usage() {
        let response = BedrockStreamingResponse {
            usage: None,
            terminal_metadata: None,
        };

        // Zero-valued usage is rig's documented sentinel for "the provider
        // reported no usage metrics".
        assert_eq!(response.token_usage(), rig_core::completion::Usage::new());
        assert!(!response.token_usage().has_values());
    }

    #[test]
    fn bedrock_stop_reason_normalization_covers_terminal_categories() {
        for (reason, expected) in [
            (
                aws_bedrock::StopReason::EndTurn,
                CompletionFinishReason::Stop,
            ),
            (
                aws_bedrock::StopReason::MaxTokens,
                CompletionFinishReason::Length,
            ),
            (
                aws_bedrock::StopReason::ModelContextWindowExceeded,
                CompletionFinishReason::Length,
            ),
            (
                aws_bedrock::StopReason::ToolUse,
                CompletionFinishReason::ToolCalls,
            ),
            (
                aws_bedrock::StopReason::ContentFiltered,
                CompletionFinishReason::ContentFilter,
            ),
        ] {
            let metadata = terminal_metadata_from_stop_reason(&reason);
            assert_eq!(metadata.reason(), expected);
            assert_eq!(metadata.raw_reason(), Some(reason.as_str()));
        }

        let unknown = aws_bedrock::StopReason::from("future_stop_reason");
        let metadata = terminal_metadata_from_stop_reason(&unknown);
        assert_eq!(metadata.reason(), CompletionFinishReason::Unknown);
        assert_eq!(metadata.raw_reason(), Some("future_stop_reason"));
    }

    #[test]
    fn test_get_completion_metadata_trait() {
        let response = BedrockStreamingResponse {
            usage: Some(BedrockUsage {
                input_tokens: 448,
                output_tokens: 68,
                total_tokens: 516,
                cache_read_input_tokens: Some(80),
                cache_write_input_tokens: Some(20),
            }),
            terminal_metadata: None,
        };

        // Test that GetCompletionMetadata trait is properly implemented
        assert_eq!(
            response.token_usage(),
            rig_core::completion::Usage {
                input_tokens: 448,
                output_tokens: 68,
                total_tokens: 516,
                cached_input_tokens: 80,
                cache_creation_input_tokens: 20,
                tool_use_prompt_tokens: 0,
                reasoning_tokens: 0,
            }
        );
    }

    #[test]
    fn test_bedrock_usage_serde() {
        let usage = BedrockUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
            cache_read_input_tokens: Some(25),
            cache_write_input_tokens: Some(5),
        };

        // Test serialization
        let json = serde_json::to_string(&usage).expect("Should serialize");
        assert!(json.contains("\"input_tokens\":100"));
        assert!(json.contains("\"output_tokens\":50"));
        assert!(json.contains("\"total_tokens\":150"));

        // Test deserialization
        let deserialized: BedrockUsage = serde_json::from_str(&json).expect("Should deserialize");
        assert_eq!(deserialized.input_tokens, usage.input_tokens);
        assert_eq!(deserialized.output_tokens, usage.output_tokens);
        assert_eq!(deserialized.total_tokens, usage.total_tokens);
        assert_eq!(
            deserialized.cache_read_input_tokens,
            usage.cache_read_input_tokens
        );
        assert_eq!(
            deserialized.cache_write_input_tokens,
            usage.cache_write_input_tokens
        );
    }

    #[test]
    fn test_bedrock_streaming_response_serde() {
        let response = BedrockStreamingResponse {
            usage: Some(BedrockUsage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
                cache_read_input_tokens: Some(30),
                cache_write_input_tokens: Some(15),
            }),
            terminal_metadata: None,
        };

        // Test serialization
        let json = serde_json::to_string(&response).expect("Should serialize");
        assert!(json.contains("\"input_tokens\":200"));

        // Test deserialization
        let deserialized: BedrockStreamingResponse =
            serde_json::from_str(&json).expect("Should deserialize");
        assert!(deserialized.usage.is_some());
        let usage = deserialized.usage.unwrap();
        assert_eq!(usage.input_tokens, 200);
        assert_eq!(usage.output_tokens, 75);
        assert_eq!(usage.total_tokens, 275);
        assert_eq!(usage.cache_read_input_tokens, Some(30));
        assert_eq!(usage.cache_write_input_tokens, Some(15));
    }

    #[test]
    fn test_reasoning_state_default() {
        // Test that ReasoningState defaults are correct
        let state = ReasoningState::default();
        assert_eq!(state.content, "");
        assert_eq!(state.signature, None);
    }

    #[test]
    fn test_reasoning_state_accumulate_content() {
        // Test accumulating content in ReasoningState
        let mut state = ReasoningState::default();
        state.content.push_str("First chunk");
        state.content.push_str(" Second chunk");
        state.content.push_str(" Third chunk");

        assert_eq!(state.content, "First chunk Second chunk Third chunk");
        assert_eq!(state.signature, None);
    }

    #[test]
    fn test_reasoning_state_with_signature() {
        // Test ReasoningState with signature
        let mut state = ReasoningState::default();
        state.content.push_str("Reasoning content");
        state.signature = Some("test_signature_456".to_string());

        assert_eq!(state.content, "Reasoning content");
        assert_eq!(state.signature, Some("test_signature_456".to_string()));
    }

    #[test]
    fn test_reasoning_state_empty_content() {
        // Test that ReasoningState can have empty content
        let state = ReasoningState {
            signature: Some("signature_only".to_string()),
            ..Default::default()
        };

        assert_eq!(state.content, "");
        assert!(state.signature.is_some());
    }

    #[test]
    fn test_tool_call_state_default() {
        // Test that ToolCallState defaults are correct
        let state = ToolCallState::default();
        assert_eq!(state.name, "");
        assert_eq!(state.id, "");
        assert_eq!(state.input_json, "");
    }

    #[test]
    fn test_tool_call_state_accumulate_json() {
        // Test accumulating JSON input in ToolCallState
        let mut state = ToolCallState {
            name: "my_tool".to_string(),
            id: "tool_123".to_string(),
            internal_call_id: rig_core::id::generate(),
            input_json: String::new(),
        };

        state.input_json.push_str("{\"arg1\":");
        state.input_json.push_str("\"value1\"");
        state.input_json.push('}');

        assert_eq!(state.name, "my_tool");
        assert_eq!(state.id, "tool_123");
        assert_eq!(state.input_json, "{\"arg1\":\"value1\"}");
    }

    #[test]
    fn test_tool_call_state_empty_accumulation() {
        let state = ToolCallState {
            name: "test_tool".to_string(),
            id: "tool_abc".to_string(),
            internal_call_id: rig_core::id::generate(),
            input_json: String::new(),
        };

        assert_eq!(state.name, "test_tool");
        assert_eq!(state.id, "tool_abc");
        assert!(state.input_json.is_empty());
    }

    #[test]
    fn test_tool_call_state_single_chunk() {
        let mut state = ToolCallState {
            name: "get_weather".to_string(),
            id: "call_123".to_string(),
            internal_call_id: rig_core::id::generate(),
            input_json: String::new(),
        };

        state.input_json.push_str("{\"location\":\"Paris\"}");

        assert_eq!(state.input_json, "{\"location\":\"Paris\"}");
    }

    #[test]
    fn test_tool_call_state_multiple_small_chunks() {
        let mut state = ToolCallState {
            name: "search".to_string(),
            id: "call_xyz".to_string(),
            internal_call_id: rig_core::id::generate(),
            input_json: String::new(),
        };

        // Simulate multiple small chunks arriving
        let chunks = vec!["{", "\"q", "uery", "\":", "\"R", "ust", "\"}"];

        for chunk in chunks {
            state.input_json.push_str(chunk);
        }

        assert_eq!(state.input_json, "{\"query\":\"Rust\"}");
    }

    #[test]
    fn test_tool_call_state_complex_json_accumulation() {
        let mut state = ToolCallState {
            name: "analyze_data".to_string(),
            id: "call_456".to_string(),
            internal_call_id: rig_core::id::generate(),
            input_json: String::new(),
        };

        // Simulate accumulating a complex nested JSON
        state.input_json.push_str("{\"data\":{");
        state.input_json.push_str("\"values\":[1,2,3],");
        state
            .input_json
            .push_str("\"metadata\":{\"source\":\"api\"}");
        state.input_json.push_str("}}");

        assert_eq!(
            state.input_json,
            "{\"data\":{\"values\":[1,2,3],\"metadata\":{\"source\":\"api\"}}}"
        );

        // Verify it's valid JSON
        let parsed: serde_json::Value =
            serde_json::from_str(&state.input_json).expect("Should parse as valid JSON");
        assert!(parsed.is_object());
    }

    #[test]
    fn test_reasoning_state_accumulation() {
        let mut state = ReasoningState::default();

        state.content.push_str("First, ");
        state.content.push_str("I need to ");
        state.content.push_str("analyze the problem.");

        assert_eq!(state.content, "First, I need to analyze the problem.");
        assert!(state.signature.is_none());
    }

    #[test]
    fn test_reasoning_state_with_signature_accumulation() {
        let mut state = ReasoningState::default();

        state.content.push_str("Reasoning content here");
        state.signature = Some("sig_part1".to_string());

        // Simulate signature being built up (in practice it comes in one chunk)
        if let Some(ref mut sig) = state.signature {
            sig.push_str("_part2");
        }

        assert_eq!(state.content, "Reasoning content here");
        assert_eq!(state.signature, Some("sig_part1_part2".to_string()));
    }

    #[test]
    fn finalize_reasoning_with_content_and_signature_emits_text_block() {
        let state = ReasoningState {
            content: "I am thinking".to_string(),
            signature: Some("sig-abc".to_string()),
        };

        let choice = finalize_reasoning(state).expect("should emit reasoning");
        match choice {
            RawStreamingChoice::Reasoning { id, content } => {
                assert!(id.is_none());
                match content {
                    ReasoningContent::Text { text, signature } => {
                        assert_eq!(text, "I am thinking");
                        assert_eq!(signature.as_deref(), Some("sig-abc"));
                    }
                    other => panic!("expected ReasoningContent::Text, got {:?}", other),
                }
            }
            _ => panic!("expected RawStreamingChoice::Reasoning"),
        }
    }

    #[test]
    fn finalize_reasoning_signature_only_still_emits_block() {
        // Adaptive-thinking on Bedrock can produce a Signature delta with no
        // accompanying non-empty Text delta. Previously this was silently
        // dropped, losing the signature and breaking next-turn replay.
        let state = ReasoningState {
            content: String::new(),
            signature: Some("sig-only".to_string()),
        };

        let choice =
            finalize_reasoning(state).expect("should emit reasoning for signature-only state");
        match choice {
            RawStreamingChoice::Reasoning { content, .. } => match content {
                ReasoningContent::Text { text, signature } => {
                    assert!(text.is_empty());
                    assert_eq!(signature.as_deref(), Some("sig-only"));
                }
                other => panic!("expected ReasoningContent::Text, got {:?}", other),
            },
            _ => panic!("expected RawStreamingChoice::Reasoning"),
        }
    }

    #[test]
    fn finalize_reasoning_content_only_still_emits_block() {
        let state = ReasoningState {
            content: "thoughts without sig".to_string(),
            signature: None,
        };

        let choice =
            finalize_reasoning(state).expect("should emit reasoning for content-only state");
        match choice {
            RawStreamingChoice::Reasoning { content, .. } => match content {
                ReasoningContent::Text { text, signature } => {
                    assert_eq!(text, "thoughts without sig");
                    assert!(signature.is_none());
                }
                other => panic!("expected ReasoningContent::Text, got {:?}", other),
            },
            _ => panic!("expected RawStreamingChoice::Reasoning"),
        }
    }

    #[test]
    fn finalize_reasoning_both_empty_emits_nothing() {
        let state = ReasoningState::default();
        assert!(finalize_reasoning(state).is_none());
    }
}
