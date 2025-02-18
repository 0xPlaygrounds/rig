use crate::types::completion_request::AwsCompletionRequest;
use crate::{completion::CompletionModel, types::errors::AwsSdkConverseStreamError};
use async_stream::stream;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig::{
    completion::CompletionError,
    streaming::{StreamingChoice, StreamingCompletionModel, StreamingResult},
};

#[derive(Default)]
struct ToolCallState {
    name: String,
    id: String,
    input_json: String,
}

impl StreamingCompletionModel for CompletionModel {
    async fn stream(
        &self,
        completion_request: rig::completion::CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let request = AwsCompletionRequest(completion_request);

        let mut converse_builder = self
            .client
            .aws_client
            .converse_stream()
            .model_id(self.model.as_str());

        let tool_config = request.tools_config()?;
        let prompt_with_history = request.prompt_with_history()?;
        converse_builder = converse_builder
            .set_additional_model_request_fields(request.additional_params())
            .set_inference_config(request.inference_config())
            .set_tool_config(tool_config)
            .set_system(request.system_prompt())
            .set_messages(Some(prompt_with_history));

        let response = converse_builder
            .send()
            .await
            .map_err(|sdk_error| AwsSdkConverseStreamError(sdk_error).into())
            .map_err(|e: CompletionError| e)?;

        Ok(Box::pin(stream! {
            let mut current_tool_call: Option<ToolCallState> = None;
            let mut stream = response.stream;
            while let Ok(Some(output)) = stream.recv().await {
                match output {
                    aws_bedrock::ConverseStreamOutput::ContentBlockDelta(event) => {
                        let delta = event.delta.ok_or(CompletionError::ProviderError("The delta for a content block is missing".into()))?;
                        match delta {
                            aws_bedrock::ContentBlockDelta::Text(text) => {
                                if current_tool_call.is_none() {
                                    yield Ok(StreamingChoice::Message(text))
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
                    aws_bedrock::ConverseStreamOutput::ContentBlockStop(_) => {
                        if let Some(tool_call) = current_tool_call.take() {
                            let json_str = if tool_call.input_json.is_empty() {
                                "{}"
                            } else {
                                &tool_call.input_json
                            };
                            match serde_json::from_str(json_str) {
                                Ok(json_value) => {
                                    yield Ok(StreamingChoice::ToolCall(
                                        tool_call.name,
                                        tool_call.id,
                                        json_value,
                                    ));
                                },
                                Err(e) => {
                                    yield Err(CompletionError::from(e));
                                }
                            }

                        }
                    },
                    // aws_bedrock::ConverseStreamOutput::MessageStart(message_start_event) => todo!(),
                    // aws_bedrock::ConverseStreamOutput::MessageStop(message_stop_event) => todo!(),
                    // aws_bedrock::ConverseStreamOutput::Metadata(converse_stream_metadata_event) => todo!(),
                    _ => {}
                }
            }
        }))
    }
}
