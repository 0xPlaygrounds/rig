use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{Text, ToolResult, ToolResultContent, UserContent},
    OneOrMany,
};

use super::{document::RigDocument, image::RigImage, tool::RigToolResultContent};

pub struct RigUserContent(pub UserContent);

impl TryFrom<aws_bedrock::ContentBlock> for RigUserContent {
    type Error = CompletionError;

    fn try_from(value: aws_bedrock::ContentBlock) -> Result<Self, Self::Error> {
        match value {
            aws_bedrock::ContentBlock::Text(text) => {
                Ok(RigUserContent(UserContent::Text(Text { text })))
            }
            aws_bedrock::ContentBlock::ToolResult(tool_result) => {
                let tool_result_contents = tool_result
                    .content
                    .into_iter()
                    .map(|tool| tool.try_into())
                    .collect::<Result<Vec<RigToolResultContent>, _>>()?
                    .into_iter()
                    .map(|rt| rt.0)
                    .collect::<Vec<ToolResultContent>>();

                let tool_results = OneOrMany::many(tool_result_contents).map_err(|_| {
                    CompletionError::ProviderError("ToolResult returned invalid response".into())
                })?;
                Ok(RigUserContent(UserContent::ToolResult(ToolResult {
                    id: tool_result.tool_use_id,
                    content: tool_results,
                })))
            }
            aws_bedrock::ContentBlock::Document(document) => {
                let doc: RigDocument = document.try_into()?;
                Ok(RigUserContent(UserContent::Document(doc.0)))
            }
            aws_bedrock::ContentBlock::Image(image) => {
                let image: RigImage = image.try_into()?;
                Ok(RigUserContent(UserContent::Image(image.0)))
            }
            _ => Err(CompletionError::ProviderError(
                "ToolResultContentBlock contains unsupported variant".into(),
            )),
        }
    }
}

impl TryFrom<RigUserContent> for Vec<aws_bedrock::ContentBlock> {
    type Error = CompletionError;

    fn try_from(value: RigUserContent) -> Result<Self, Self::Error> {
        match value.0 {
            UserContent::Text(text) => Ok(vec![aws_bedrock::ContentBlock::Text(text.text)]),
            UserContent::ToolResult(tool_result) => {
                let builder = aws_bedrock::ToolResultBlock::builder()
                    .tool_use_id(tool_result.id)
                    .set_content(Some(
                        tool_result
                            .content
                            .into_iter()
                            .map(|tool| RigToolResultContent(tool).try_into())
                            .collect::<Result<Vec<aws_bedrock::ToolResultContentBlock>, _>>()?,
                    ))
                    .build()
                    .map_err(|e| CompletionError::ProviderError(e.to_string()))?;
                Ok(vec![aws_bedrock::ContentBlock::ToolResult(builder)])
            }
            UserContent::Image(image) => {
                let image = RigImage(image).try_into()?;
                Ok(vec![aws_bedrock::ContentBlock::Image(image)])
            }
            UserContent::Document(document) => {
                let doc = RigDocument(document).try_into()?;
                // AWS documentations: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
                // In the content field of the Message object, you must also include a text field with a prompt related to the document.
                Ok(vec![
                    aws_bedrock::ContentBlock::Text("Use provided document".to_string()),
                    aws_bedrock::ContentBlock::Document(doc),
                ])
            }
            UserContent::Audio(_) => Err(CompletionError::ProviderError(
                "Audio is not supported".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::user_content::RigUserContent;
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use rig::{
        completion::CompletionError,
        message::{ToolResultContent, UserContent},
        OneOrMany,
    };

    #[test]
    fn aws_content_block_to_user_content() {
        let cb = aws_bedrock::ContentBlock::Text("42".into());
        let user_content: Result<RigUserContent, _> = cb.try_into();
        assert_eq!(user_content.is_ok(), true);
        let content = match user_content.unwrap().0 {
            rig::message::UserContent::Text(text) => Ok(text),
            _ => Err("Invalid content type"),
        };
        assert_eq!(content.is_ok(), true);
        assert_eq!(content.unwrap().text, "42")
    }

    #[test]
    fn aws_content_block_tool_to_user_content() {
        let cb = aws_bedrock::ContentBlock::ToolResult(
            aws_bedrock::ToolResultBlock::builder()
                .tool_use_id("123")
                .content(aws_bedrock::ToolResultContentBlock::Text("content".into()))
                .build()
                .unwrap(),
        );
        let user_content: Result<RigUserContent, _> = cb.try_into();
        assert_eq!(user_content.is_ok(), true);
        let content = match user_content.unwrap().0 {
            rig::message::UserContent::ToolResult(tool_result) => Ok(tool_result),
            _ => Err("Invalid content type"),
        };
        assert_eq!(content.is_ok(), true);
        let content = content.unwrap();
        assert_eq!(content.id, "123");
        assert_eq!(
            content.content,
            OneOrMany::one(ToolResultContent::Text("content".into()))
        )
    }

    #[test]
    fn aws_unsupported_content_block_to_user_content() {
        let cb = aws_bedrock::ContentBlock::GuardContent(
            aws_bedrock::GuardrailConverseContentBlock::Text(
                aws_bedrock::GuardrailConverseTextBlock::builder()
                    .text("stuff")
                    .build()
                    .unwrap(),
            ),
        );
        let user_content: Result<RigUserContent, _> = cb.try_into();
        assert_eq!(user_content.is_ok(), false);
        assert_eq!(
            user_content.err().unwrap().to_string(),
            CompletionError::ProviderError(
                "ToolResultContentBlock contains unsupported variant".into()
            )
            .to_string()
        )
    }

    #[test]
    fn user_content_to_aws_content_block() {
        let uc = RigUserContent(UserContent::Text("txt".into()));
        let aws_content_blocks: Result<Vec<aws_bedrock::ContentBlock>, _> = uc.try_into();
        assert_eq!(aws_content_blocks.is_ok(), true);
        let aws_content_blocks = aws_content_blocks.unwrap();
        assert_eq!(
            aws_content_blocks,
            vec![aws_bedrock::ContentBlock::Text("txt".into())]
        );
    }
}
