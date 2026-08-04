use crate::types::assistant_content::{PROVIDER_NAME, map_stop_reason};
use crate::types::completion_request::AwsCompletionRequest;
use crate::types::converse_output::StopReason;
use crate::{
    completion::{CompletionModel, resolve_request_model},
    types::errors::{AwsSdkConverseStreamError, converse_stream_output_completion_error},
};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::streaming::StreamingCompletionResponse;
use rig_core::telemetry::{CompletionOperation, CompletionSpanBuilder, SpanCombinator};
use rig_core::{
    completion::CompletionError,
    message::ReasoningContent,
    streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing_futures::Instrument;

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<BedrockUsage>,
    /// Bedrock's own `stopReason` from the terminal `MessageStop` event, when
    /// the stream reported one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
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

impl From<&BedrockStreamingResponse> for rig_core::completion::Usage {
    fn from(response: &BedrockStreamingResponse) -> Self {
        response
            .usage
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

/// Accumulated per-stream state for [`process_event`].
///
/// In-flight tool calls are keyed by Bedrock's own `contentBlockIndex`: the
/// Converse stream indexes every content block, and a message may open several
/// tool-use blocks before it stops, so a single "current" slot would let a
/// later block silently overwrite an earlier one.
#[derive(Default)]
struct StreamState {
    tool_calls: HashMap<i32, ToolCallState>,
    current_reasoning: Option<ReasoningState>,
    final_stop_reason: Option<StopReason>,
}

/// Convert an accumulated [`ToolCallState`] into the tool-call item to yield.
///
/// An empty accumulated input means a tool with no parameters; malformed JSON
/// is surfaced as an error item rather than silently dropped, so the terminal
/// can never report tool use whose calls the consumer never saw.
fn finalize_tool_call(
    tool_call: ToolCallState,
) -> Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError> {
    let tool_input = if tool_call.input_json.is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(tool_call.input_json.as_str()).map_err(|err| {
            CompletionError::ResponseError(format!(
                "tool call `{}` arrived with malformed JSON input: {err}",
                tool_call.name
            ))
        })?
    };
    Ok(RawStreamingChoice::ToolCall(
        RawStreamingToolCall::new(tool_call.id, tool_call.name, tool_input)
            .with_internal_call_id(tool_call.internal_call_id),
    ))
}

/// Handle one Converse stream event, returning the items to yield in order.
///
/// Kept as a plain function over [`StreamState`] so the event bookkeeping can
/// be unit-tested without an AWS event receiver.
fn process_event(
    state: &mut StreamState,
    output: aws_bedrock::ConverseStreamOutput,
) -> Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>> {
    let mut items = Vec::new();
    match output {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
            let Some(delta) = event.delta else {
                tracing::warn!("skipping ContentBlockDelta with a missing delta");
                return items;
            };
            match delta {
                aws_bedrock::ContentBlockDelta::Text(text) => {
                    items.push(Ok(RawStreamingChoice::Message(text)));
                }
                aws_bedrock::ContentBlockDelta::ToolUse(tool) => {
                    if let Some(tool_call) = state.tool_calls.get_mut(&event.content_block_index) {
                        let delta = tool.input().to_string();
                        tool_call.input_json.push_str(&delta);

                        // Emit the delta so UI can show progress
                        items.push(Ok(RawStreamingChoice::ToolCallDelta {
                            id: tool_call.id.clone(),
                            internal_call_id: tool_call.internal_call_id.clone(),
                            content: ToolCallDeltaContent::Delta(delta),
                        }));
                    }
                }
                aws_bedrock::ContentBlockDelta::ReasoningContent(reasoning) => match reasoning {
                    aws_bedrock::ReasoningContentBlockDelta::Text(text) => {
                        state
                            .current_reasoning
                            .get_or_insert_with(ReasoningState::default)
                            .content
                            .push_str(text.as_str());

                        if !text.is_empty() {
                            items.push(Ok(RawStreamingChoice::ReasoningDelta {
                                reasoning: text.clone(),
                                id: None,
                            }));
                        }
                    }
                    aws_bedrock::ReasoningContentBlockDelta::Signature(signature) => {
                        state
                            .current_reasoning
                            .get_or_insert_with(ReasoningState::default)
                            .signature = Some(signature.clone());
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        aws_bedrock::ConverseStreamOutput::ContentBlockStart(event) => {
            let Some(start) = event.start else {
                tracing::warn!("skipping ContentBlockStart with no data");
                return items;
            };
            match start {
                aws_bedrock::ContentBlockStart::ToolUse(tool_use) => {
                    let internal_call_id = rig_core::id::generate();
                    state.tool_calls.insert(
                        event.content_block_index,
                        ToolCallState {
                            name: tool_use.name.clone(),
                            id: tool_use.tool_use_id.clone(),
                            internal_call_id: internal_call_id.clone(),
                            input_json: String::new(),
                        },
                    );
                    items.push(Ok(RawStreamingChoice::ToolCallDelta {
                        id: tool_use.tool_use_id,
                        internal_call_id,
                        content: ToolCallDeltaContent::Name(tool_use.name),
                    }));
                }
                _ => items.push(Err(CompletionError::ProviderError(
                    "Stream is empty".into(),
                ))),
            }
        }
        aws_bedrock::ConverseStreamOutput::ContentBlockStop(event) => {
            if let Some(reasoning_state) = state.current_reasoning.take()
                && let Some(choice) = finalize_reasoning(reasoning_state)
            {
                items.push(Ok(choice));
            }
            // A closed tool-use block is complete: finalize and emit it here,
            // mirroring the reasoning finalize above, so every call in a
            // multi-tool-call message reaches the consumer.
            if let Some(tool_call) = state.tool_calls.remove(&event.content_block_index) {
                items.push(finalize_tool_call(tool_call));
            }
        }
        aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => {
            // Remember Bedrock's own terminal reason so the final
            // record can report it; an unmapped SDK variant is kept
            // verbatim rather than dropped.
            state.final_stop_reason = Some(
                StopReason::try_from(message_stop_event.stop_reason.clone()).unwrap_or_else(|_| {
                    StopReason::Unknown(crate::types::converse_output::UnknownVariantValue(
                        message_stop_event.stop_reason.as_str().to_owned(),
                    ))
                }),
            );
            // Tool calls normally flush at their ContentBlockStop; drain any
            // stragglers here defensively (in block order) so a stream that
            // omits the stop event still delivers every call. Other stop
            // reasons (including MaxTokens) are genuine provider terminals,
            // not errors: the Metadata path emits the terminal record with
            // the mapped finish reason via `final_stop_reason`.
            let mut remaining: Vec<(i32, ToolCallState)> = state.tool_calls.drain().collect();
            remaining.sort_by_key(|(index, _)| *index);
            for (_, tool_call) in remaining {
                items.push(finalize_tool_call(tool_call));
            }
        }
        aws_bedrock::ConverseStreamOutput::Metadata(metadata_event) => {
            // Extract usage information from metadata; a missing usage still
            // yields a terminal record so the stream ends with a FinalResponse.
            let final_response = BedrockStreamingResponse {
                usage: metadata_event.usage.map(|usage| BedrockUsage {
                    input_tokens: usage.input_tokens,
                    output_tokens: usage.output_tokens,
                    total_tokens: usage.total_tokens,
                    cache_read_input_tokens: usage.cache_read_input_tokens,
                    cache_write_input_tokens: usage.cache_write_input_tokens,
                }),
                stop_reason: state.final_stop_reason.clone(),
            };
            items.push(Ok(RawStreamingChoice::FinalResponse(final_response)));
        }
        _ => {}
    }
    items
}

impl CompletionModel {
    /// Open a stream whose terminal record stays Bedrock's own response type.
    pub async fn raw_stream(
        &self,
        completion_request: rig_core::completion::CompletionRequest,
    ) -> Result<rig_core::streaming::RawStreamingResult<BedrockStreamingResponse>, CompletionError>
    {
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
            let mut state = StreamState::default();
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
                for item in process_event(&mut state, output) {
                    if let Ok(RawStreamingChoice::FinalResponse(final_response)) = &item {
                        span.record_token_usage(&final_response.into());
                    }
                    yield item;
                }
            }
        }.instrument(span));

        Ok(stream)
    }

    /// Open a stream normalized to rig's terminal record. Delegates to
    /// [`CompletionModel::raw_stream`] — one request either way.
    pub(crate) async fn stream(
        &self,
        completion_request: rig_core::completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse, CompletionError> {
        let raw = self.raw_stream(completion_request).await?;
        let normalized = rig_core::streaming::normalize_stream(raw, |response| {
            let usage = (&response).into();
            let finish_reason = response.stop_reason.as_ref().map(map_stop_reason);
            Ok(rig_core::streaming::StreamFinal::new(PROVIDER_NAME, usage)
                .with_optional_finish_reason(finish_reason))
        });

        Ok(StreamingCompletionResponse::stream(
            PROVIDER_NAME,
            normalized,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            stop_reason: None,
        };

        assert_eq!(
            rig_core::completion::Usage::from(&response),
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
            stop_reason: None,
        };

        // Zero-valued usage is rig's documented sentinel for "the provider
        // reported no usage metrics".
        assert_eq!(
            rig_core::completion::Usage::from(&response),
            rig_core::completion::Usage::new()
        );
        assert!(!rig_core::completion::Usage::from(&response).has_values());
    }

    #[test]
    fn test_streaming_response_normalizes_usage() {
        let response = BedrockStreamingResponse {
            usage: Some(BedrockUsage {
                input_tokens: 448,
                output_tokens: 68,
                total_tokens: 516,
                cache_read_input_tokens: Some(80),
                cache_write_input_tokens: Some(20),
            }),
            stop_reason: None,
        };

        // The streaming response normalizes into rig's usage record.
        assert_eq!(
            rig_core::completion::Usage::from(&response),
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
            stop_reason: None,
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

    fn tool_start_event(index: i32, id: &str, name: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockStart(
            aws_bedrock::ContentBlockStartEvent::builder()
                .content_block_index(index)
                .start(aws_bedrock::ContentBlockStart::ToolUse(
                    aws_bedrock::ToolUseBlockStart::builder()
                        .tool_use_id(id)
                        .name(name)
                        .build()
                        .expect("tool use start should build"),
                ))
                .build()
                .expect("content block start should build"),
        )
    }

    fn tool_delta_event(index: i32, input: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::ToolUse(
                    aws_bedrock::ToolUseBlockDelta::builder()
                        .input(input)
                        .build()
                        .expect("tool use delta should build"),
                ))
                .build()
                .expect("content block delta should build"),
        )
    }

    fn text_delta_event(index: i32, text: &str) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockDelta(
            aws_bedrock::ContentBlockDeltaEvent::builder()
                .content_block_index(index)
                .delta(aws_bedrock::ContentBlockDelta::Text(text.to_string()))
                .build()
                .expect("content block delta should build"),
        )
    }

    fn block_stop_event(index: i32) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::ContentBlockStop(
            aws_bedrock::ContentBlockStopEvent::builder()
                .content_block_index(index)
                .build()
                .expect("content block stop should build"),
        )
    }

    fn message_stop_event(reason: aws_bedrock::StopReason) -> aws_bedrock::ConverseStreamOutput {
        aws_bedrock::ConverseStreamOutput::MessageStop(
            aws_bedrock::MessageStopEvent::builder()
                .stop_reason(reason)
                .build()
                .expect("message stop should build"),
        )
    }

    /// Run a sequence of events through [`process_event`] with fresh state,
    /// returning every item the stream would yield, plus the final state.
    fn run_events(
        events: Vec<aws_bedrock::ConverseStreamOutput>,
    ) -> (
        Vec<Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>>,
        StreamState,
    ) {
        let mut state = StreamState::default();
        let mut items = Vec::new();
        for event in events {
            items.extend(process_event(&mut state, event));
        }
        (items, state)
    }

    fn emitted_tool_calls(
        items: &[Result<RawStreamingChoice<BedrockStreamingResponse>, CompletionError>],
    ) -> Vec<&RawStreamingToolCall> {
        items
            .iter()
            .filter_map(|item| match item {
                Ok(RawStreamingChoice::ToolCall(call)) => Some(call),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn parallel_tool_calls_all_emitted_with_tool_use_terminal() {
        // Two tool-use blocks in one message: both must survive, and the
        // latched stop reason must map to a tool-use terminal.
        let (items, state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\":"),
            tool_delta_event(0, "\"Paris\"}"),
            block_stop_event(0),
            tool_start_event(1, "call_b", "get_time"),
            tool_delta_event(1, "{\"zone\":\"UTC\"}"),
            block_stop_event(1),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        let calls = emitted_tool_calls(&items);
        assert_eq!(calls.len(), 2, "both parallel tool calls must be emitted");
        let first = calls.first().expect("first call");
        assert_eq!(first.id, "call_a");
        assert_eq!(first.name, "get_weather");
        assert_eq!(first.arguments, serde_json::json!({"location": "Paris"}));
        let second = calls.get(1).expect("second call");
        assert_eq!(second.id, "call_b");
        assert_eq!(second.name, "get_time");
        assert_eq!(second.arguments, serde_json::json!({"zone": "UTC"}));

        // The terminal reports tool use with the calls actually delivered.
        assert_eq!(state.final_stop_reason, Some(StopReason::ToolUse));
        assert_eq!(
            map_stop_reason(&StopReason::ToolUse),
            rig_core::completion::FinishReason::ToolCalls
        );
        assert!(items.iter().all(|item| item.is_ok()));
    }

    #[test]
    fn tool_call_flushes_at_content_block_stop() {
        // The call must not wait for MessageStop: closing the block emits it.
        let (items, state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{}"),
            block_stop_event(0),
        ]);

        assert_eq!(emitted_tool_calls(&items).len(), 1);
        assert!(state.tool_calls.is_empty(), "state must be cleared at stop");
    }

    #[test]
    fn message_stop_flushes_stragglers_missing_a_block_stop() {
        // Defensive path: a stream that omits ContentBlockStop still delivers
        // every accumulated call at MessageStop, in block order.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\":\"Paris\"}"),
            tool_start_event(1, "call_b", "get_time"),
            tool_delta_event(1, "{\"zone\":\"UTC\"}"),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        let calls = emitted_tool_calls(&items);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls.first().expect("first call").id, "call_a");
        assert_eq!(calls.get(1).expect("second call").id, "call_b");
    }

    #[test]
    fn text_after_closed_tool_block_is_delivered() {
        // A text block following a closed tool-use block used to be discarded
        // because the single tool slot was never cleared.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{}"),
            block_stop_event(0),
            text_delta_event(1, "Checking the weather now."),
            block_stop_event(1),
            message_stop_event(aws_bedrock::StopReason::EndTurn),
        ]);

        let texts: Vec<&str> = items
            .iter()
            .filter_map(|item| match item {
                Ok(RawStreamingChoice::Message(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["Checking the weather now."]);
        assert_eq!(emitted_tool_calls(&items).len(), 1);
    }

    #[test]
    fn malformed_tool_json_surfaces_an_error_item() {
        // Malformed accumulated input must not be silently dropped while the
        // terminal still claims tool use: the consumer gets an error item.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "get_weather"),
            tool_delta_event(0, "{\"location\": not-json"),
            block_stop_event(0),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        assert!(emitted_tool_calls(&items).is_empty());
        assert!(
            items.iter().any(|item| matches!(
                item,
                Err(CompletionError::ResponseError(msg)) if msg.contains("get_weather")
            )),
            "malformed tool JSON must yield an error item"
        );
    }

    #[test]
    fn empty_tool_input_becomes_empty_object() {
        // A tool with no parameters streams no input deltas at all.
        let (items, _state) = run_events(vec![
            tool_start_event(0, "call_a", "ping"),
            block_stop_event(0),
            message_stop_event(aws_bedrock::StopReason::ToolUse),
        ]);

        let calls = emitted_tool_calls(&items);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls.first().expect("call").arguments,
            serde_json::json!({})
        );
    }
}
