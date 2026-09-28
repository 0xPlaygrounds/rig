use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::{Text, UserContent};

use super::{
    converse_output::ContentBlock, document::RigDocument, image::RigImage,
    tool::RigToolResultContent,
};

pub struct RigUserContent(pub UserContent);

impl TryFrom<ContentBlock> for RigUserContent {
    type Error = ProviderError;

    fn try_from(value: ContentBlock) -> Result<Self, Self::Error> {
        match value {
            ContentBlock::Text(text) => Ok(RigUserContent(UserContent::Text(Text::new(text)))),
            // Bedrock's wire correlates results by `toolUseId` only and never
            // carries the tool name a result needs.
            ContentBlock::ToolResult(_) => Err(ProviderError::Provider(
                "AWS Bedrock returned a tool result, which names no tool".into(),
            )),
            ContentBlock::Document(document) => {
                let doc: RigDocument = document.try_into()?;
                Ok(RigUserContent(UserContent::Document(doc.0)))
            }
            ContentBlock::Image(image) => {
                let image: RigImage = image.try_into()?;
                Ok(RigUserContent(UserContent::Image(image.0)))
            }
            _ => Err(ProviderError::Provider(
                "ToolResultContentBlock contains unsupported variant".into(),
            )),
        }
    }
}

impl TryFrom<RigUserContent> for Vec<aws_bedrock::ContentBlock> {
    type Error = ProviderError;

    fn try_from(value: RigUserContent) -> Result<Self, Self::Error> {
        match value.0 {
            UserContent::Text(text) => Ok(vec![aws_bedrock::ContentBlock::Text(text.text)]),
            UserContent::ToolResult(tool_result) => {
                let builder = aws_bedrock::ToolResultBlock::builder()
                    .tool_use_id(tool_result.call.wire().into_owned())
                    .set_content(Some(
                        tool_result
                            .content
                            .into_iter()
                            .map(|tool| RigToolResultContent(tool).try_into())
                            .collect::<Result<Vec<aws_bedrock::ToolResultContentBlock>, _>>()?,
                    ))
                    .build()
                    .map_err(|e| ProviderError::Provider(e.to_string()))?;
                Ok(vec![aws_bedrock::ContentBlock::ToolResult(builder)])
            }
            UserContent::Image(image) => {
                let image = RigImage(image).try_into()?;
                Ok(vec![aws_bedrock::ContentBlock::Image(image)])
            }
            UserContent::Document(document) => {
                let doc = RigDocument(document).try_into()?;
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
}

#[cfg(test)]
mod tests;
