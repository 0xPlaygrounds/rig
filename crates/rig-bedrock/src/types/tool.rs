use aws_sdk_bedrockruntime::types as aws_bedrock;

use super::{image, json};
use rig_core::error::ProviderError;
use rig_core::message::ToolResultContent;
use serde_json::Value;

/// The Converse tool-result block for `content`; blank text has none.
pub(crate) fn to_aws(
    content: ToolResultContent,
) -> Result<Option<aws_bedrock::ToolResultContentBlock>, ProviderError> {
    match content {
        ToolResultContent::Text(text) if text.text.trim().is_empty() => Ok(None),
        ToolResultContent::Text(text) => {
            Ok(Some(aws_bedrock::ToolResultContentBlock::Text(text.text)))
        }
        ToolResultContent::Image(image) => {
            let image = image::to_aws(image)?;
            Ok(Some(aws_bedrock::ToolResultContentBlock::Image(image)))
        }
        ToolResultContent::Json { value } => {
            // Object-only tool-result schemas require a wrapper for other
            // JSON shapes without converting structured data to text.
            let value = match value {
                Value::Object(_) => value,
                value => serde_json::json!({ "result": value }),
            };
            Ok(Some(aws_bedrock::ToolResultContentBlock::Json(
                json::to_document(value),
            )))
        }
    }
}

#[cfg(test)]
mod tests;
