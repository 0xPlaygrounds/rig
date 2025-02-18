use crate::{
    completion::CompletionModel,
    types::{errors::AwsSdkConverseStreamError, json::AwsDocument, message::MessageWithPrompt},
};
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
        mut completion_request: rig::completion::CompletionRequest,
    ) -> Result<StreamingResult, CompletionError> {
        let mut full_history = Vec::new();
        full_history.append(&mut completion_request.chat_history);
        full_history.push(completion_request.prompt_with_context());

        let prompt_with_history = full_history
            .into_iter()
            .map(|message| {
                MessageWithPrompt {
                    message,
                    prompt: completion_request.preamble.clone(),
                }
                .try_into()
            })
            .collect::<Result<Vec<aws_bedrock::Message>, _>>()?;

        let mut converse_builder = self
            .client
            .aws_client
            .converse_stream()
            .model_id(self.model.as_str());

        let mut inference_configuration = aws_bedrock::InferenceConfiguration::builder();

        if let Some(params) = completion_request.additional_params {
            let doc: AwsDocument = params.into();
            converse_builder = converse_builder.set_additional_model_request_fields(Some(doc.0));
        }

        if let Some(temperature) = completion_request.temperature {
            inference_configuration =
                inference_configuration.set_temperature(Some(temperature as f32));
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            inference_configuration =
                inference_configuration.set_max_tokens(Some(max_tokens as i32));
        }

        converse_builder =
            converse_builder.set_inference_config(Some(inference_configuration.build()));

        let mut tools = vec![];
        for tool_definition in completion_request.tools.iter() {
            let doc: AwsDocument = tool_definition.parameters.clone().into();
            let schema = aws_bedrock::ToolInputSchema::Json(doc.0);
            let tool = aws_bedrock::Tool::ToolSpec(
                aws_bedrock::ToolSpecification::builder()
                    .name(tool_definition.name.clone())
                    .set_description(Some(tool_definition.description.clone()))
                    .set_input_schema(Some(schema))
                    .build()
                    .map_err(|e| CompletionError::RequestError(e.into()))?,
            );
            tools.push(tool);
        }

        if !tools.is_empty() {
            let config = aws_bedrock::ToolConfiguration::builder()
                .set_tools(Some(tools))
                .build()
                .map_err(|e| CompletionError::RequestError(e.into()))?;

            converse_builder = converse_builder.set_tool_config(Some(config));
        }

        if let Some(system_prompt) = completion_request.preamble {
            converse_builder =
                converse_builder.set_system(Some(vec![aws_bedrock::SystemContentBlock::Text(
                    system_prompt,
                )]));
        }

        let response = converse_builder
            .set_messages(Some(prompt_with_history))
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
