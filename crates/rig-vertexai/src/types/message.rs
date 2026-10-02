use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::error::ProviderError;
use rig_core::message::{
    DocumentSourceKind, Image, ImageMediaType, Message, MimeType, Text, ToolResultContent,
    UserContent,
};
use rig_core::providers::gemini::completion::gemini_api_types::assistant_part;
use std::collections::HashSet;

/// The Vertex AI `Content` for a non-system `message`. System messages
/// travel in `system_instruction` and are rejected here.
pub(crate) fn content_from_message(
    message: Message,
) -> Result<vertexai::model::Content, ProviderError> {
    match message {
        Message::System { .. } => Err(ProviderError::Provider(
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
                                    response_parts
                                        .push(vertex_tool_result_image_part(image, &display_name)?);
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

                        // Function responses correlate by name, not call ID.
                        let function_name = tool_result.name.clone();
                        let function_response = vertexai::model::FunctionResponse::new()
                            .set_name(function_name)
                            .set_response(response_struct)
                            .set_parts(response_parts);

                        Ok(vertexai::model::Part::new().set_function_response(function_response))
                    }
                    _ => Err(ProviderError::Provider(format!(
                        "Unsupported user content type: {user_content:?}"
                    ))),
                })
                .collect();

            let parts = parts?;
            Ok(vertexai::model::Content::new()
                .set_role("user")
                .set_parts(parts))
        }
        Message::Assistant(turn) => {
            // Vertex function calls and responses carry no id, and the SDK's
            // byte signatures cannot hold Google's placeholder spelling.
            let parts = turn
                .content
                .iter()
                .filter_map(|block| assistant_part(block, None, false).transpose())
                .map(|part| serde_json::from_value(part?).map_err(ProviderError::request))
                .collect::<Result<Vec<vertexai::model::Part>, ProviderError>>()?;
            Ok(vertexai::model::Content::new()
                .set_role("model")
                .set_parts(parts))
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
) -> Result<vertexai::model::FunctionResponsePart, ProviderError> {
    let media_type = image.media_type.as_ref().ok_or_else(|| {
        ProviderError::request("Media type for tool-result image is required for Vertex AI")
    })?;
    match media_type {
        ImageMediaType::JPEG | ImageMediaType::PNG | ImageMediaType::WEBP => {}
        unsupported => {
            return Err(ProviderError::request(format!(
                "Unsupported Vertex AI tool-result image media type {unsupported:?}; \
                     expected JPEG, PNG, or WEBP"
            )));
        }
    }
    let mime_type = media_type.to_mime_type();

    let data = match &image.data {
        DocumentSourceKind::Base64(data) => BASE64.decode(data.as_bytes()).map_err(|error| {
            ProviderError::request(format!("Invalid base64 tool-result image data: {error}"))
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
            return Err(ProviderError::request(format!(
                "Unsupported Vertex AI tool-result image source: {unsupported}"
            )));
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

#[cfg(test)]
mod tests;
