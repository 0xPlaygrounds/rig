use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{Text, ToolResult, UserContent},
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
                let tool_results =
                    OneOrMany::many(tool_result.content.into_iter().filter_map(|tool| {
                        tool.try_into().ok().map(|rt: RigToolResultContent| rt.0)
                    }))
                    .map_err(|_| {
                        CompletionError::ProviderError(
                            "ToolResult returned invalid response".into(),
                        )
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

pub struct UserContentWithPrompt {
    pub user_content: UserContent,
    pub prompt: Option<String>,
}

impl TryFrom<UserContentWithPrompt> for Vec<aws_bedrock::ContentBlock> {
    type Error = CompletionError;

    fn try_from(value: UserContentWithPrompt) -> Result<Self, Self::Error> {
        match value.user_content {
            UserContent::Text(text) => Ok(vec![aws_bedrock::ContentBlock::Text(text.text)]),
            UserContent::ToolResult(tool_result) => {
                let builder = aws_bedrock::ToolResultBlock::builder()
                    .tool_use_id(tool_result.id)
                    .set_content(Some(
                        tool_result
                            .content
                            .into_iter()
                            .filter_map(|tool| RigToolResultContent(tool).try_into().ok())
                            .collect(),
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
                // AWS documentations: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference-call.html
                // In the content field of the Message object, you must also include a text field with a prompt related to the document.

                if let Some(prompt) = value.prompt.filter(|p| p.len() > 1) {
                    let doc = RigDocument(document).try_into()?;
                    Ok(vec![
                        aws_bedrock::ContentBlock::Text(prompt),
                        aws_bedrock::ContentBlock::Document(doc),
                    ])
                } else {
                    Err(CompletionError::ProviderError(
                        "Document upload required system prompt".into(),
                    ))
                }
            }
            UserContent::Audio(_) => Err(CompletionError::ProviderError(
                "Audio is not supported".into(),
            )),
        }
    }
}
