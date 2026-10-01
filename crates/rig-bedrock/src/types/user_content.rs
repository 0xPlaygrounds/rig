use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::UserContent;

use super::{document, image, tool};

/// The Converse content blocks for one piece of user content.
pub(crate) fn to_aws(
    content: UserContent,
) -> Result<Vec<aws_bedrock::ContentBlock>, ProviderError> {
    match content {
        UserContent::Text(text) => Ok(vec![aws_bedrock::ContentBlock::Text(text.text)]),
        UserContent::ToolResult(tool_result) => {
            let builder = aws_bedrock::ToolResultBlock::builder()
                .tool_use_id(tool_result.call.wire().into_owned())
                .set_content(Some(
                    tool_result
                        .content
                        .into_iter()
                        .map(tool::to_aws)
                        .collect::<Result<Vec<aws_bedrock::ToolResultContentBlock>, _>>()?,
                ))
                .build()
                .map_err(|e| ProviderError::Provider(e.to_string()))?;
            Ok(vec![aws_bedrock::ContentBlock::ToolResult(builder)])
        }
        UserContent::Image(image) => {
            let image = image::to_aws(image)?;
            Ok(vec![aws_bedrock::ContentBlock::Image(image)])
        }
        UserContent::Document(document) => {
            let doc = document::to_aws(document)?;
            // Converse requires accompanying prompt text for document blocks.
            Ok(vec![
                aws_bedrock::ContentBlock::Text("Use provided document".to_string()),
                aws_bedrock::ContentBlock::Document(doc),
            ])
        }
        UserContent::Audio(_) => Err(ProviderError::Provider("Audio is not supported".into())),
        UserContent::Video(_) => Err(ProviderError::Provider("Video is not supported".into())),
    }
}

#[cfg(test)]
mod tests;
