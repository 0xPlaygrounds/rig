use aws_sdk_bedrockruntime::types as aws_bedrock;

use super::{converse_output::ToolResultContentBlock, image::RigImage, json::AwsDocument};
use rig_core::error::ProviderError;
use rig_core::message::{Text, ToolResultContent};
use serde_json::Value;

pub struct RigToolResultContent(pub ToolResultContent);

impl TryFrom<RigToolResultContent> for aws_bedrock::ToolResultContentBlock {
    type Error = ProviderError;

    fn try_from(value: RigToolResultContent) -> Result<Self, Self::Error> {
        match value.0 {
            ToolResultContent::Text(text) => {
                Ok(aws_bedrock::ToolResultContentBlock::Text(text.text))
            }
            ToolResultContent::Image(image) => {
                let image = RigImage(image).try_into()?;
                Ok(aws_bedrock::ToolResultContentBlock::Image(image))
            }
            ToolResultContent::Json { value } => {
                // Object-only tool-result schemas require a wrapper for other
                // JSON shapes without converting structured data to text.
                let value = match value {
                    Value::Object(_) => value,
                    value => serde_json::json!({ "result": value }),
                };
                let document: AwsDocument = value.into();
                Ok(aws_bedrock::ToolResultContentBlock::Json(document.0))
            }
        }
    }
}

impl TryFrom<ToolResultContentBlock> for RigToolResultContent {
    type Error = ProviderError;

    fn try_from(value: ToolResultContentBlock) -> Result<Self, Self::Error> {
        match value {
            ToolResultContentBlock::Image(image) => {
                let image: RigImage = image.try_into()?;
                Ok(RigToolResultContent(ToolResultContent::Image(image.0)))
            }
            ToolResultContentBlock::Json(value) => {
                Ok(RigToolResultContent(ToolResultContent::Json { value }))
            }
            ToolResultContentBlock::Text(text) => Ok(RigToolResultContent(
                ToolResultContent::Text(Text::new(text)),
            )),
            _ => Err(ProviderError::Provider(
                "ToolResultContentBlock contains unsupported variant".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
