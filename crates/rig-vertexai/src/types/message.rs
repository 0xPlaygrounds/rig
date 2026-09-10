use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionError;
use rig_core::message::{
    AssistantContent, DocumentSourceKind, Image, ImageMediaType, Message, MimeType, Text,
    ToolResultContent, UserContent,
};
use std::collections::HashSet;

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
                            // Vertex carries media in `parts` and locates it in
                            // the structured response through display-name
                            // references, preserving canonical block order.
                            let mut outputs = Vec::new();
                            let mut response_parts = Vec::new();
                            let mut reserved_display_names = HashSet::new();
                            for content in tool_result.content.iter() {
                                if let ToolResultContent::Json { value } = content {
                                    collect_json_ref_names(value, &mut reserved_display_names);
                                }
                            }
                            let mut image_index = 0;

                            for content in tool_result.content.iter() {
                                match content {
                                    ToolResultContent::Text(Text { text, .. }) => {
                                        outputs.push(serde_json::Value::String(text.clone()));
                                    }
                                    ToolResultContent::Json { value } => {
                                        outputs.push(value.clone());
                                    }
                                    ToolResultContent::Image(image) => {
                                        let display_name = loop {
                                            let candidate =
                                                format!("rig_tool_result_image_{image_index}");
                                            image_index += 1;
                                            if reserved_display_names.insert(candidate.clone()) {
                                                break candidate;
                                            }
                                        };
                                        response_parts.push(vertex_tool_result_image_part(
                                            image,
                                            &display_name,
                                        )?);
                                        outputs.push(serde_json::json!({ "$ref": display_name }));
                                    }
                                }
                            }

                            let output_value = match outputs.as_slice() {
                                [single] => single.clone(),
                                _ => serde_json::Value::Array(outputs),
                            };

                            let mut response_struct = serde_json::Map::new();
                            response_struct.insert("output".to_string(), output_value);

                            // `functionResponse.name` is the executed
                            // function's name — required data on the result.
                            let function_name = tool_result.name.clone();
                            let function_response = vertexai::model::FunctionResponse::new()
                                .set_name(function_name)
                                .set_response(response_struct)
                                .set_parts(response_parts);

                            Ok(vertexai::model::Part::new()
                                .set_function_response(function_response))
                        }
                        _ => Err(CompletionError::ProviderError(format!(
                            "Unsupported user content type: {user_content:?}"
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
                            let serde_json::Value::Object(struct_val) =
                                tool_call.function.arguments
                            else {
                                return Err(CompletionError::ProviderError(
                                    "Expected JSON object for Struct conversion".to_string(),
                                ));
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

fn collect_json_ref_names(value: &serde_json::Value, names: &mut HashSet<String>) {
    match value {
        serde_json::Value::Object(object) => {
            if let Some(serde_json::Value::String(name)) = object.get("$ref") {
                names.insert(name.clone());
            }
            for value in object.values() {
                collect_json_ref_names(value, names);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                collect_json_ref_names(value, names);
            }
        }
        _ => {}
    }
}

fn vertex_tool_result_image_part(
    image: &Image,
    display_name: &str,
) -> Result<vertexai::model::FunctionResponsePart, CompletionError> {
    let media_type = image.media_type.as_ref().ok_or_else(|| {
        CompletionError::RequestError(
            "Media type for tool-result image is required for Vertex AI".into(),
        )
    })?;
    match media_type {
        ImageMediaType::JPEG | ImageMediaType::PNG | ImageMediaType::WEBP => {}
        unsupported => {
            return Err(CompletionError::RequestError(
                format!(
                    "Unsupported Vertex AI tool-result image media type {unsupported:?}; \
                     expected JPEG, PNG, or WEBP"
                )
                .into(),
            ));
        }
    }
    let mime_type = media_type.to_mime_type();

    let data = match &image.data {
        DocumentSourceKind::Base64(data) => BASE64.decode(data.as_bytes()).map_err(|error| {
            CompletionError::RequestError(
                format!("Invalid base64 tool-result image data: {error}").into(),
            )
        })?,
        DocumentSourceKind::Raw(data) => data.clone(),
        DocumentSourceKind::Url(url) => {
            return Ok(vertexai::model::FunctionResponsePart::new().set_file_data(
                vertexai::model::FunctionResponseFileData::new()
                    .set_mime_type(mime_type)
                    .set_file_uri(url.clone())
                    .set_display_name(display_name),
            ));
        }
        unsupported => {
            return Err(CompletionError::RequestError(
                format!("Unsupported Vertex AI tool-result image source: {unsupported}").into(),
            ));
        }
    };

    Ok(
        vertexai::model::FunctionResponsePart::new().set_inline_data(
            vertexai::model::FunctionResponseBlob::new()
                .set_mime_type(mime_type)
                .set_data(data)
                .set_display_name(display_name),
        ),
    )
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
mod tests;
