use crate::types::json_utils;
use google_cloud_aiplatform_v1 as vertexai;
use rig::completion::CompletionError;
use rig::message::{AssistantContent, Message, Text, ToolResultContent, UserContent};

pub struct RigMessage(pub Message);

impl TryFrom<RigMessage> for vertexai::model::Content {
    type Error = CompletionError;

    fn try_from(value: RigMessage) -> Result<Self, Self::Error> {
        match value.0 {
            Message::User { content } => {
                let parts: Result<Vec<vertexai::model::Part>, _> = content
                    .into_iter()
                    .map(|user_content| match user_content {
                        UserContent::Text(Text { text }) => {
                            Ok(vertexai::model::Part::new().set_text(text))
                        }
                        UserContent::ToolResult(tool_result) => {
                            let response_struct = if tool_result.content.len() == 1 {
                                match tool_result.content.iter().next() {
                                    Some(ToolResultContent::Text(Text { text })) => {
                                        serde_json::json!({ "output": text })
                                    }
                                    _ => {
                                        serde_json::json!({ "output": "Tool executed successfully" })
                                    }
                                }
                            } else {
                                serde_json::json!({ "output": "Multiple results" })
                            };

                            let struct_val = json_utils::json_to_struct(response_struct)?;
                            
                            let function_response = vertexai::model::FunctionResponse::new()
                                .set_name(tool_result.id.clone())
                                .set_response(struct_val);

                            Ok(vertexai::model::Part::new().set_function_response(function_response))
                        }
                        _ => Err(CompletionError::ProviderError(
                            format!("Unsupported user content type: {:?}", user_content),
                        )),
                    })
                    .collect();

                let parts = parts?;
                Ok(vertexai::model::Content::new()
                    .set_role("user")
                    .set_parts(parts))
            }
            Message::Assistant { content, .. } => {
                let parts: Result<Vec<vertexai::model::Part>, _> = content
                    .into_iter()
                    .map(|assistant_content| match assistant_content {
                        AssistantContent::Text(Text { text }) => {
                            Ok(vertexai::model::Part::new().set_text(text))
                        }
                        AssistantContent::ToolCall(tool_call) => {
                            let struct_val = json_utils::json_to_struct(tool_call.function.arguments)?;
                            
                            let function_call = vertexai::model::FunctionCall::new()
                                .set_name(tool_call.function.name.clone())
                                .set_args(struct_val);

                            Ok(vertexai::model::Part::new().set_function_call(function_call))
                        }
                        _ => Err(CompletionError::ProviderError(
                            format!("Unsupported assistant content type: {:?}", assistant_content),
                        )),
                    })
                    .collect();

                let parts = parts?;
                Ok(vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts(parts))
            }
        }
    }
}

