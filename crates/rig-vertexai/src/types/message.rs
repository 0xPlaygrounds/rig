use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionError;
use rig_core::message::{
    AssistantContent, DocumentSourceKind, Image, ImageMediaType, Message, MimeType, Text,
    ToolResultContent, UserContent,
};

pub struct RigMessage(pub Message);

impl TryFrom<RigMessage> for vertexai::model::Content {
    type Error = CompletionError;

    fn try_from(value: RigMessage) -> Result<Self, Self::Error> {
        match value.0 {
            Message::System { .. } => Err(CompletionError::ProviderError(
                "System messages must be sent via Vertex AI system_instruction".to_string(),
            )),
            Message::User { content } => {
                let parts: Result<Vec<vertexai::model::Part>, _> = content
                    .into_iter()
                    .map(|user_content| match user_content {
                        UserContent::Text(Text { text, .. }) => {
                            Ok(vertexai::model::Part::new().set_text(text))
                        }
                        UserContent::ToolResult(tool_result) => {
                            // Vertex AI accepts native JSON under `output`. Keep a singleton
                            // scalar/object as-is and use an array only when preserving the
                            // order of multiple Rig content blocks.
                            let outputs = tool_result
                                .content
                                .into_iter()
                                .map(|content| match content {
                                    ToolResultContent::Text(Text { text, .. }) => {
                                        Ok(serde_json::Value::String(text))
                                    }
                                    ToolResultContent::Json { value } => Ok(value),
                                    ToolResultContent::Image(_) => Err(message_conversion_error(
                                        "Vertex AI does not support images in tool results; return text or JSON instead",
                                    )),
                                })
                                .collect::<Result<Vec<_>, CompletionError>>()?;

                            let output_value = match outputs.as_slice() {
                                [single] => single.clone(),
                                _ => serde_json::Value::Array(outputs),
                            };

                            let mut response_struct = serde_json::Map::new();
                            response_struct.insert("output".to_string(), output_value);

                            let function_response = vertexai::model::FunctionResponse::new()
                                .set_name(tool_result.id.clone())
                                .set_response(response_struct);

                            Ok(vertexai::model::Part::new()
                                .set_function_response(function_response))
                        }
                        _ => Err(CompletionError::ProviderError(format!(
                            "Unsupported user content type: {:?}",
                            user_content
                        ))),
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
                        AssistantContent::Text(Text { text, .. }) => {
                            Ok(vertexai::model::Part::new().set_text(text))
                        }
                        AssistantContent::Image(image) => vertex_assistant_image_part(image),
                        AssistantContent::ToolCall(tool_call) => {
                            let struct_val = match tool_call.function.arguments {
                                serde_json::Value::Object(map) => map,
                                _ => {
                                    return Err(CompletionError::ProviderError(
                                        "Expected JSON object for Struct conversion".to_string(),
                                    ));
                                }
                            };

                            let function_call = vertexai::model::FunctionCall::new()
                                .set_name(tool_call.function.name.clone())
                                .set_args(struct_val);

                            let mut part =
                                vertexai::model::Part::new().set_function_call(function_call);

                            // Echo back the Gemini `thoughtSignature` captured on the read side
                            // (base64 → bytes). Required by thinking models on every follow-up turn.
                            // A malformed signature is dropped (with a warning) rather than failing
                            // the whole turn — one bad byte must not kill every other tool call.
                            if let Some(signature) = &tool_call.signature {
                                match BASE64.decode(signature.as_bytes()) {
                                    Ok(bytes) => part = part.set_thought_signature(bytes),
                                    Err(err) => tracing::warn!(
                                        %err,
                                        tool = %tool_call.function.name,
                                        "Failed to base64-decode tool call thought_signature; \
                                         dropping it for this turn"
                                    ),
                                }
                            }

                            Ok(part)
                        }
                        AssistantContent::Reasoning(reasoning) => {
                            let mut part = vertexai::model::Part::new()
                                .set_text(reasoning.display_text())
                                .set_thought(true);

                            if let Some(signature) = reasoning.first_signature() {
                                match BASE64.decode(signature.as_bytes()) {
                                    Ok(bytes) => part = part.set_thought_signature(bytes),
                                    Err(err) => tracing::warn!(
                                        %err,
                                        "Failed to base64-decode reasoning thought_signature; \
                                         dropping it for this turn"
                                    ),
                                }
                            }

                            Ok(part)
                        }
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

fn message_conversion_error(message: impl Into<String>) -> CompletionError {
    rig_core::message::MessageError::ConversionError(message.into()).into()
}

fn vertex_assistant_image_part(image: Image) -> Result<vertexai::model::Part, CompletionError> {
    let media_type = image.media_type.ok_or_else(|| {
        CompletionError::RequestError(
            "Media type for assistant image is required for Vertex AI".into(),
        )
    })?;

    match media_type {
        ImageMediaType::JPEG
        | ImageMediaType::PNG
        | ImageMediaType::WEBP
        | ImageMediaType::HEIC
        | ImageMediaType::HEIF => {}
        unsupported => {
            return Err(CompletionError::RequestError(
                format!("Unsupported Vertex AI assistant image media type {unsupported:?}").into(),
            ));
        }
    }

    let DocumentSourceKind::Base64(data) = image.data else {
        return Err(CompletionError::RequestError(
            "Vertex AI assistant images must use base64 data".into(),
        ));
    };

    let data = BASE64.decode(data.as_bytes()).map_err(|err| {
        CompletionError::RequestError(format!("Invalid base64 assistant image data: {err}").into())
    })?;

    Ok(vertexai::model::Part::new().set_inline_data(
        vertexai::model::Blob::new()
            .set_mime_type(media_type.to_mime_type())
            .set_data(data),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::completion_response::VertexGenerateContentOutput;
    use google_cloud_aiplatform_v1 as vertexai;
    use rig_core::OneOrMany;
    use rig_core::completion::CompletionResponse;
    use rig_core::message::{Message, Text, ToolResult, ToolResultContent};

    fn vertex_tool_result(
        content: OneOrMany<ToolResultContent>,
    ) -> Result<vertexai::model::Content, CompletionError> {
        RigMessage(Message::User {
            content: OneOrMany::one(UserContent::ToolResult(ToolResult {
                id: "lookup".to_string(),
                call_id: None,
                content,
            })),
        })
        .try_into()
    }

    fn function_response(content: &vertexai::model::Content) -> &vertexai::model::FunctionResponse {
        content.parts[0]
            .function_response()
            .expect("function response part")
    }

    fn conversion_error_message(error: CompletionError) -> String {
        let CompletionError::RequestError(source) = error else {
            panic!("expected request error wrapping MessageError, got {error:?}");
        };
        let error = source
            .downcast_ref::<rig_core::message::MessageError>()
            .expect("request error should preserve MessageError");
        match error {
            rig_core::message::MessageError::ConversionError(message) => message.clone(),
        }
    }

    #[test]
    fn test_user_text_message_conversion() {
        let message = Message::User {
            content: OneOrMany::one(rig_core::message::UserContent::Text(Text::new(
                "Hello".to_string(),
            ))),
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
            content: OneOrMany::one(AssistantContent::Text(Text::new("Hi there".to_string()))),
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
    fn test_assistant_image_response_round_trips_through_history_in_order() {
        let raw_image = vec![0, 1, 2, 255];
        let response = vertexai::model::GenerateContentResponse::new().set_candidates([
            vertexai::model::Candidate::new().set_content(
                vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts([
                        vertexai::model::Part::new().set_text("before"),
                        vertexai::model::Part::new().set_inline_data(
                            vertexai::model::Blob::new()
                                .set_mime_type("image/png")
                                .set_data(raw_image.clone()),
                        ),
                        vertexai::model::Part::new().set_text("after"),
                    ]),
            ),
        ]);
        let response: CompletionResponse<VertexGenerateContentOutput> =
            VertexGenerateContentOutput(response)
                .try_into()
                .expect("image response should convert");

        let content: vertexai::model::Content = RigMessage(Message::Assistant {
            id: None,
            content: response.choice,
        })
        .try_into()
        .expect("assistant history image should convert");

        assert_eq!(content.parts.len(), 3);
        assert_eq!(content.parts[0].text().map(String::as_str), Some("before"));
        let image = content.parts[1]
            .inline_data()
            .expect("middle part should be an inline image");
        assert_eq!(image.mime_type, "image/png");
        assert_eq!(image.data.as_ref(), raw_image.as_slice());
        assert_eq!(content.parts[2].text().map(String::as_str), Some("after"));
    }

    #[test]
    fn test_assistant_image_history_rejects_invalid_or_unsupported_input() {
        let cases = [
            (
                AssistantContent::image_base64(BASE64.encode([1]), None, None),
                "Media type",
            ),
            (
                AssistantContent::image_base64(BASE64.encode([1]), Some(ImageMediaType::GIF), None),
                "Unsupported",
            ),
            (
                AssistantContent::image_base64("not valid base64", Some(ImageMediaType::PNG), None),
                "Invalid base64",
            ),
        ];

        for (image, expected_message) in cases {
            let result: Result<vertexai::model::Content, CompletionError> =
                RigMessage(Message::Assistant {
                    id: None,
                    content: OneOrMany::one(image),
                })
                .try_into();
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("invalid assistant image must fail"),
            };
            assert!(matches!(error, CompletionError::RequestError(_)));
            assert!(error.to_string().contains(expected_message));
        }
    }

    #[test]
    fn test_assistant_tool_call_message_conversion() {
        use rig_core::message::{ToolCall, ToolFunction};
        let tool_call = ToolCall::new(
            "add".to_string(),
            ToolFunction::new(
                "add".to_string(),
                serde_json::json!({
                    "x": 5,
                    "y": 3
                }),
            ),
        );

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
    fn test_assistant_tool_call_echoes_thought_signature() {
        use rig_core::message::{ToolCall, ToolFunction};
        let raw = b"\x00\x01\x02thinking-sig\xff";
        let tool_call = ToolCall::new(
            "add".to_string(),
            ToolFunction::new("add".to_string(), serde_json::json!({"x": 5})),
        )
        .with_signature(Some(BASE64.encode(raw)));
        let content: vertexai::model::Content = RigMessage(Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
        })
        .try_into()
        .unwrap();
        assert_eq!(content.parts[0].thought_signature.as_ref(), raw.as_slice());
    }

    #[test]
    fn test_assistant_tool_call_malformed_signature_is_dropped_not_fatal() {
        // A malformed signature must not abort the whole turn — it is dropped with a warning.
        use rig_core::message::{ToolCall, ToolFunction};
        let tool_call = ToolCall::new(
            "add".to_string(),
            ToolFunction::new("add".to_string(), serde_json::json!({"x": 5})),
        )
        .with_signature(Some("!!! not base64 !!!".to_string()));
        let content: vertexai::model::Content = RigMessage(Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::ToolCall(tool_call)),
        })
        .try_into()
        .expect("malformed signature should not fail the conversion");
        assert_eq!(content.parts.len(), 1);
        assert!(content.parts[0].thought_signature.is_empty());
        assert!(content.parts[0].function_call().is_some());
    }

    #[test]
    fn test_assistant_reasoning_echoes_thought_signature() {
        let raw = b"\x00\x01\x02thinking-text-sig\xff";
        let reasoning = rig_core::message::Reasoning::new_with_signature(
            "thinking text",
            Some(BASE64.encode(raw)),
        );

        let content: vertexai::model::Content = RigMessage(Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
        })
        .try_into()
        .unwrap();

        assert_eq!(content.parts.len(), 1);
        assert_eq!(
            content.parts[0].text().map(String::as_str),
            Some("thinking text")
        );
        assert!(content.parts[0].thought);
        assert_eq!(content.parts[0].thought_signature.as_ref(), raw.as_slice());
    }

    #[test]
    fn test_assistant_reasoning_malformed_signature_is_dropped_not_fatal() {
        let reasoning = rig_core::message::Reasoning::new_with_signature(
            "thinking text",
            Some("!!! not base64 !!!".to_string()),
        );

        let content: vertexai::model::Content = RigMessage(Message::Assistant {
            id: None,
            content: OneOrMany::one(AssistantContent::Reasoning(reasoning)),
        })
        .try_into()
        .expect("malformed signature should not fail the conversion");

        assert_eq!(content.parts.len(), 1);
        assert!(content.parts[0].thought);
        assert!(content.parts[0].thought_signature.is_empty());
    }

    #[test]
    fn test_user_tool_result_message_conversion() {
        let content =
            vertex_tool_result(OneOrMany::one(ToolResultContent::text(r#"{"answer":8}"#)))
                .expect("literal text tool result should convert");
        assert_eq!(content.role.as_str(), "user");
        assert_eq!(content.parts.len(), 1);

        let function_response = function_response(&content);
        assert_eq!(function_response.name.as_str(), "lookup");
        assert_eq!(
            function_response
                .response
                .as_ref()
                .expect("response")
                .get("output"),
            Some(&serde_json::json!(r#"{"answer":8}"#))
        );
    }

    #[test]
    fn test_user_tool_result_preserves_native_json_singleton() {
        let value = serde_json::json!({
            "answer": 8,
            "metadata": { "exact": true }
        });
        let content = vertex_tool_result(OneOrMany::one(ToolResultContent::json(value.clone())))
            .expect("structured JSON tool result should convert");

        assert_eq!(
            function_response(&content)
                .response
                .as_ref()
                .expect("response")
                .get("output"),
            Some(&value)
        );
    }

    #[test]
    fn test_user_tool_result_preserves_text_json_order() {
        let content = vertex_tool_result(
            OneOrMany::many([
                ToolResultContent::text("before"),
                ToolResultContent::json(serde_json::json!({ "step": 2 })),
                ToolResultContent::text("after"),
            ])
            .expect("non-empty tool result"),
        )
        .expect("ordered mixed tool result should convert");

        assert_eq!(
            function_response(&content)
                .response
                .as_ref()
                .expect("response")
                .get("output"),
            Some(&serde_json::json!(["before", { "step": 2 }, "after"]))
        );
    }

    #[test]
    fn test_user_tool_result_images_return_conversion_errors() {
        let error = vertex_tool_result(OneOrMany::one(ToolResultContent::image_base64(
            "image-data",
            Some(ImageMediaType::PNG),
            None,
        )))
        .expect_err("Vertex AI tool results cannot represent images");

        assert!(conversion_error_message(error).contains("does not support images"));
    }
}
