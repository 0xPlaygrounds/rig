use crate::types::completion_request::AwsCompletionRequest;
use crate::{completion::CompletionModel, types::errors::AwsSdkConverseStreamError};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig::completion::GetTokenUsage;
use rig::streaming::StreamingCompletionResponse;
use rig::{completion::CompletionError, streaming::RawStreamingChoice};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockStreamingResponse {
    pub usage: Option<BedrockUsage>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct BedrockUsage {
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub total_tokens: i32,
}

impl GetTokenUsage for BedrockStreamingResponse {
    fn token_usage(&self) -> Option<rig::completion::Usage> {
        self.usage.as_ref().map(|u| rig::completion::Usage {
            input_tokens: u.input_tokens as u64,
            output_tokens: u.output_tokens as u64,
            total_tokens: u.total_tokens as u64,
        })
    }
}

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    input_json: String,
}

#[derive(Default)]
struct ReasoningState {
    content: String,
    signature: Option<String>,
}

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: rig::completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<BedrockStreamingResponse>, CompletionError> {
        let request = AwsCompletionRequest(completion_request);

        let mut converse_builder = self
            .client
            .get_inner()
            .await
            .converse_stream()
            .model_id(self.model.as_str());

        let tool_config = request.tools_config()?;
        let prompt_with_history = request.messages()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt())
            .set_messages(Some(prompt_with_history));

        let response = converse_builder.send().await.map_err(|sdk_error| {
            Into::<CompletionError>::into(AwsSdkConverseStreamError(sdk_error))
        })?;

        let stream = Box::pin(stream! {
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut current_reasoning: Option<ReasoningState> = None;
            let mut stream = response.stream;
            while let Ok(Some(output)) = stream.recv().await {
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
                                    tool_call.input_json.push_str(tool.input());
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
                                            yield Ok(RawStreamingChoice::Reasoning {
                                                reasoning: text.clone(),
                                                id: None,
                                                signature: None,
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
                                current_tool_call = Some(ToolCallState {
                                    name: tool_use.name,
                                    id: tool_use.tool_use_id,
                                    input_json: String::new(),
                                });
                            },
                            _ => yield Err(CompletionError::ProviderError("Stream is empty".into()))
                        }
                    },
                    aws_bedrock::ConverseStreamOutput::ContentBlockStop(_event) => {
                        if let Some(reasoning_state) = current_reasoning.take()
                            && !reasoning_state.content.is_empty() {
                                yield Ok(RawStreamingChoice::Reasoning {
                                    reasoning: reasoning_state.content,
                                    id: None,
                                    signature: reasoning_state.signature,
                                })
                            }
                    },
                    aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => {
                        match message_stop_event.stop_reason {
                            aws_bedrock::StopReason::ToolUse => {
                                if let Some(tool_call) = current_tool_call.take() {
                                    let tool_input = serde_json::from_str(tool_call.input_json.as_str())?;
                                    yield Ok(RawStreamingChoice::ToolCall {
                                        name: tool_call.name,
                                        call_id: None,
                                        id: tool_call.id,
                                        arguments: tool_input
                                    });
                                } else {
                                    yield Err(CompletionError::ProviderError("Failed to call tool".into()))
                                }
                            }
                            aws_bedrock::StopReason::MaxTokens => {
                                yield Err(CompletionError::ProviderError("Exceeded max tokens".into()))
                            }
                            _ => {}
                        }
                    },
                    aws_bedrock::ConverseStreamOutput::Metadata(metadata_event) => {
                        // Extract usage information from metadata
                        if let Some(usage) = metadata_event.usage {
                            yield Ok(RawStreamingChoice::FinalResponse(BedrockStreamingResponse {
                                usage: Some(BedrockUsage {
                                    input_tokens: usage.input_tokens,
                                    output_tokens: usage.output_tokens,
                                    total_tokens: usage.total_tokens,
                                }),
                            }));
                        }
                    },
                    _ => {}
                }
            }
        });

        Ok(StreamingCompletionResponse::stream(stream))
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
            }),
        };

        let rig_usage = response.token_usage();
        assert!(rig_usage.is_some());

        let usage = rig_usage.unwrap();
        assert_eq!(usage.input_tokens, 200);
        assert_eq!(usage.output_tokens, 75);
        assert_eq!(usage.total_tokens, 275);
    }

    #[test]
    fn test_bedrock_streaming_response_without_usage() {
        let response = BedrockStreamingResponse { usage: None };

        let rig_usage = response.token_usage();
        assert!(rig_usage.is_none());
    }

    #[test]
    fn test_get_token_usage_trait() {
        let response = BedrockStreamingResponse {
            usage: Some(BedrockUsage {
                input_tokens: 448,
                output_tokens: 68,
                total_tokens: 516,
            }),
        };

        // Test that GetTokenUsage trait is properly implemented
        let usage = response.token_usage().expect("Usage should be present");
        assert_eq!(usage.input_tokens, 448);
        assert_eq!(usage.output_tokens, 68);
        assert_eq!(usage.total_tokens, 516);
    }

    #[test]
    fn test_bedrock_usage_serde() {
        let usage = BedrockUsage {
            input_tokens: 100,
            output_tokens: 50,
            total_tokens: 150,
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
    }

    #[test]
    fn test_bedrock_streaming_response_serde() {
        let response = BedrockStreamingResponse {
            usage: Some(BedrockUsage {
                input_tokens: 200,
                output_tokens: 75,
                total_tokens: 275,
            }),
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
            input_json: String::new(),
        };

        state.input_json.push_str("{\"arg1\":");
        state.input_json.push_str("\"value1\"");
        state.input_json.push('}');

        assert_eq!(state.name, "my_tool");
        assert_eq!(state.id, "tool_123");
        assert_eq!(state.input_json, "{\"arg1\":\"value1\"}");
    }
}
