use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::{
    completion::CompletionError,
    message::{Text, ToolResultContent, UserContent},
};

use super::{
    converse_output::ContentBlock, document::RigDocument, image::RigImage,
    tool::RigToolResultContent,
};

pub struct RigUserContent(pub UserContent);

impl TryFrom<ContentBlock> for RigUserContent {
    type Error = CompletionError;

    fn try_from(value: ContentBlock) -> Result<Self, Self::Error> {
        match value {
            ContentBlock::Text(text) => Ok(RigUserContent(UserContent::Text(Text::new(text)))),
            ContentBlock::ToolResult(tool_result) => {
                let tool_result_contents = tool_result
                    .content
                    .into_iter()
                    .map(|tool| tool.try_into())
                    .collect::<Result<Vec<RigToolResultContent>, _>>()?
                    .into_iter()
                    .map(|rt| rt.0)
                    .collect::<Vec<ToolResultContent>>();

                let tool_results =
                    rig_core::message::require_non_empty(tool_result_contents, || {
                        CompletionError::ProviderError(
                            "ToolResult returned invalid response".into(),
                        )
                    })?;
                // Bedrock's wire correlates results by `toolUseId` only
                // and never carries the tool name; this conversion is lossy
                // for name-keyed wires.
                Ok(RigUserContent(UserContent::tool_result_from_wire(
                    tool_result.tool_use_id,
                    "",
                    tool_results,
                )))
            }
            ContentBlock::Document(document) => {
                let doc: RigDocument = document.try_into()?;
                Ok(RigUserContent(UserContent::Document(doc.0)))
            }
            ContentBlock::Image(image) => {
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
                    .tool_use_id(tool_result.wire_call_id().to_owned())
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
            UserContent::Video(_) => Err(CompletionError::ProviderError(
                "Video is not supported".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{converse_output::ContentBlock, user_content::RigUserContent};
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use rig_core::{
        completion::CompletionError,
        message::{ToolResultContent, UserContent},
    };

    /// The inbound path reads the mirror, but what Bedrock sends is the SDK
    /// block, so the tests still start there and mirror it first.
    fn mirrored(block: aws_bedrock::ContentBlock) -> ContentBlock {
        block.try_into().expect("the SDK block mirrors")
    }

    #[test]
    fn aws_content_block_to_user_content() {
        let cb = mirrored(aws_bedrock::ContentBlock::Text("42".into()));
        let user_content: Result<RigUserContent, _> = cb.try_into();
        assert!(user_content.is_ok());
        let content = match user_content.unwrap().0 {
            rig_core::message::UserContent::Text(text) => Ok(text),
            _ => Err("Invalid content type"),
        };
        assert!(content.is_ok());
        assert_eq!(content.unwrap().text, "42")
    }

    #[test]
    fn aws_content_block_tool_to_user_content() {
        let cb = mirrored(aws_bedrock::ContentBlock::ToolResult(
            aws_bedrock::ToolResultBlock::builder()
                .tool_use_id("123")
                .content(aws_bedrock::ToolResultContentBlock::Text("content".into()))
                .build()
                .unwrap(),
        ));
        let user_content: Result<RigUserContent, _> = cb.try_into();
        assert!(user_content.is_ok());
        let content = match user_content.unwrap().0 {
            rig_core::message::UserContent::ToolResult(tool_result) => Ok(tool_result),
            _ => Err("Invalid content type"),
        };
        assert!(content.is_ok());
        let content = content.unwrap();
        // Bedrock's wire id becomes the provider call id (and rig's id adopts
        // it); the wire carries no tool name, so the conversion is lossy there.
        assert_eq!(content.call, "123");
        assert_eq!(
            content.provider.as_ref().map(|p| p.call_id.as_str()),
            Some("123")
        );
        assert_eq!(content.name, "");
        assert_eq!(
            content.content,
            vec![ToolResultContent::Text("content".into())]
        )
    }

    #[test]
    fn aws_unsupported_content_block_to_user_content() {
        let cb = mirrored(aws_bedrock::ContentBlock::GuardContent(
            aws_bedrock::GuardrailConverseContentBlock::Text(
                aws_bedrock::GuardrailConverseTextBlock::builder()
                    .text("stuff")
                    .build()
                    .unwrap(),
            ),
        ));
        let user_content: Result<RigUserContent, _> = cb.try_into();
        assert!(user_content.is_err());
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
        assert!(aws_content_blocks.is_ok());
        let aws_content_blocks = aws_content_blocks.unwrap();
        assert_eq!(
            aws_content_blocks,
            vec![aws_bedrock::ContentBlock::Text("txt".into())]
        );
    }
}
