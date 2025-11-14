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

#[cfg(test)]
mod tests {
    use super::*;
    use google_cloud_aiplatform_v1 as vertexai;
    use rig::OneOrMany;
    use rig::message::{Message, Text, ToolResult, ToolResultContent};

    #[test]
    fn test_user_text_message_conversion() {
        let message = Message::User {
            content: OneOrMany::one(rig::message::UserContent::Text(Text {
                text: "Hello".to_string(),
            })),
        };

        let rig_message = RigMessage(message);
        let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

        assert!(vertex_content.is_ok());
        let content = vertex_content.unwrap();
        assert_eq!(content.role.as_str(), "user");
        assert_eq!(content.parts.len(), 1);
        assert_eq!(content.parts[0].text(), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_assistant_text_message_conversion() {
        let message = Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::Text(Text {
                text: "Hi there".to_string(),
            })),
        };

        let rig_message = RigMessage(message);
        let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

        assert!(vertex_content.is_ok());
        let content = vertex_content.unwrap();
        assert_eq!(content.role.as_str(), "model");
        assert_eq!(content.parts.len(), 1);
        assert_eq!(content.parts[0].text(), Some(&"Hi there".to_string()));
    }

    #[test]
    fn test_assistant_tool_call_message_conversion() {
        use rig::message::{ToolCall, ToolFunction};
        let tool_call = ToolCall {
            id: "add".to_string(),
            call_id: None,
            function: ToolFunction {
                name: "add".to_string(),
                arguments: serde_json::json!({
                    "x": 5,
                    "y": 3
                }),
            },
        };

        let message = Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
        };

        let rig_message = RigMessage(message);
        let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

        assert!(vertex_content.is_ok());
        let content = vertex_content.unwrap();
        assert_eq!(content.role.as_str(), "model");
        assert_eq!(content.parts.len(), 1);
        
        let function_call = content.parts[0].function_call();
        assert!(function_call.is_some());
        let function_call = function_call.unwrap();
        assert_eq!(function_call.name.as_str(), "add");
    }

    #[test]
    fn test_user_tool_result_message_conversion() {
        let tool_result = ToolResult {
            id: "add".to_string(),
            call_id: None,
            content: OneOrMany::one(ToolResultContent::Text(Text {
                text: "8".to_string(),
            })),
        };

        let message = Message::User {
            content: OneOrMany::one(rig::message::UserContent::ToolResult(tool_result)),
        };

        let rig_message = RigMessage(message);
        let vertex_content: Result<vertexai::model::Content, _> = rig_message.try_into();

        assert!(vertex_content.is_ok());
        let content = vertex_content.unwrap();
        assert_eq!(content.role.as_str(), "user");
        assert_eq!(content.parts.len(), 1);
        
        let function_response = content.parts[0].function_response();
        assert!(function_response.is_some());
        let function_response = function_response.unwrap();
        assert_eq!(function_response.name.as_str(), "add");
    }
}

