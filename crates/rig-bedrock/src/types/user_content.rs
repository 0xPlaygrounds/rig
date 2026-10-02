use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::UserContent;

use super::{document, image, tool};

/// What stands in for content left empty once blank text is skipped:
/// Converse rejects blank text and empty content.
pub(crate) const EMPTY_TEXT: &str = "<empty>";

/// The Converse content blocks for one piece of user content. Blank text has
/// none.
pub(crate) fn to_aws(
    content: UserContent,
) -> Result<Vec<aws_bedrock::ContentBlock>, ProviderError> {
    match content {
        UserContent::Text(text) if text.text.trim().is_empty() => Ok(Vec::new()),
        UserContent::Text(text) => Ok(vec![aws_bedrock::ContentBlock::Text(text.text)]),
        UserContent::ToolResult(tool_result) => {
            let mut content = Vec::new();
            for part in tool_result.content {
                content.extend(tool::to_aws(part)?);
            }
            if content.is_empty() {
                content.push(aws_bedrock::ToolResultContentBlock::Text(
                    EMPTY_TEXT.to_owned(),
                ));
            }
            // Converse reads a result with no status as a success.
            let status = tool_result
                .is_error
                .then_some(aws_bedrock::ToolResultStatus::Error);
            let builder = aws_bedrock::ToolResultBlock::builder()
                .tool_use_id(tool_result.call.wire().into_owned())
                .set_content(Some(content))
                .set_status(status)
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
