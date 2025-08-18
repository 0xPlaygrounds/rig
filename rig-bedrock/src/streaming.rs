use crate::types::completion_request::AwsCompletionRequest;
use crate::{completion::CompletionModel, types::errors::AwsSdkConverseStreamError};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig::streaming::StreamingCompletionResponse;
use rig::{completion::CompletionError, streaming::RawStreamingChoice};

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    input_json: String,
}

impl CompletionModel {
    pub(crate) async fn stream(
        &self,
        completion_request: rig::completion::CompletionRequest,
    ) -> Result<StreamingCompletionResponse<()>, CompletionError> {
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
                    _ => {}
                }
            }
        });

        Ok(StreamingCompletionResponse::stream(stream))
    }
}
