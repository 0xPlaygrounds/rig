use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{Text, ToolResultContent},
};
use serde_json::Value;

use super::{image::RigImage, json::AwsDocument};

pub struct RigToolResultContent(pub ToolResultContent);

impl TryFrom<RigToolResultContent> for aws_bedrock::ToolResultContentBlock {
    type Error = CompletionError;

    fn try_from(value: RigToolResultContent) -> Result<Self, Self::Error> {
        match value.0 {
            ToolResultContent::Text(text) => {
                Ok(aws_bedrock::ToolResultContentBlock::Text(text.text))
            }
            ToolResultContent::Image(image) => {
                let image = RigImage(image).try_into()?;
                Ok(aws_bedrock::ToolResultContentBlock::Image(image))
            }
        }
    }
}

impl TryFrom<aws_bedrock::ToolResultContentBlock> for RigToolResultContent {
    type Error = CompletionError;

    fn try_from(value: aws_bedrock::ToolResultContentBlock) -> Result<Self, Self::Error> {
        match value {
            aws_bedrock::ToolResultContentBlock::Image(image) => {
                let image: RigImage = image.try_into()?;
                Ok(RigToolResultContent(ToolResultContent::Image(image.0)))
            }
            aws_bedrock::ToolResultContentBlock::Json(document) => {
                let json: Value = AwsDocument(document).into();
                Ok(RigToolResultContent(ToolResultContent::Text(Text {
                    text: json.to_string(),
                })))
            }
            aws_bedrock::ToolResultContentBlock::Text(text) => {
                Ok(RigToolResultContent(ToolResultContent::Text(Text { text })))
            }
            _ => Err(CompletionError::ProviderError(
                "ToolResultContentBlock contains unsupported variant".into(),
            )),
        }
    }
}
