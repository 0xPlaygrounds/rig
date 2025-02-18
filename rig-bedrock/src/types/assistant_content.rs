use aws_sdk_bedrockruntime::operation::converse::ConverseOutput;
use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{AssistantContent, Text, ToolCall, ToolFunction},
    OneOrMany,
};

use crate::types::message::RigMessage;

use super::json::AwsDocument;
use rig::completion;

#[derive(Clone)]
pub struct AwsConverseOutput(pub ConverseOutput);

impl TryFrom<AwsConverseOutput> for completion::CompletionResponse<AwsConverseOutput> {
    type Error = CompletionError;

    fn try_from(value: AwsConverseOutput) -> Result<Self, Self::Error> {
        let message: RigMessage = value
            .to_owned()
            .0
            .output
            .ok_or(CompletionError::ProviderError(
                "Model didn't return any output".into(),
            ))?
            .as_message()
            .map_err(|_| {
                CompletionError::ProviderError(
                    "Failed to extract message from converse output".into(),
                )
            })?
            .to_owned()
            .try_into()?;

        let choice = match message.0 {
            completion::Message::Assistant { content } => Ok(content),
            _ => Err(CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )),
        }?;

        if let Some(tool_use) = choice.iter().find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.to_owned()),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: OneOrMany::one(AssistantContent::ToolCall(ToolCall {
                    id: tool_use.id,
                    function: ToolFunction {
                        name: tool_use.function.name,
                        arguments: tool_use.function.arguments,
                    },
                })),
                raw_response: value,
            });
        }

        Ok(completion::CompletionResponse {
            choice,
            raw_response: value,
        })
    }
}

pub struct RigAssistantContent(pub AssistantContent);

impl TryFrom<aws_bedrock::ContentBlock> for RigAssistantContent {
    type Error = CompletionError;

    fn try_from(value: aws_bedrock::ContentBlock) -> Result<Self, Self::Error> {
        match value {
            aws_bedrock::ContentBlock::Text(text) => {
                Ok(RigAssistantContent(AssistantContent::Text(Text { text })))
            }
            aws_bedrock::ContentBlock::ToolUse(call) => Ok(RigAssistantContent(
                completion::AssistantContent::tool_call(
                    &call.tool_use_id,
                    &call.name,
                    AwsDocument(call.input).into(),
                ),
            )),
            _ => Err(CompletionError::ProviderError(
                "AWS Bedrock returned unsupported ContentBlock".into(),
            )),
        }
    }
}

impl TryFrom<RigAssistantContent> for aws_bedrock::ContentBlock {
    type Error = CompletionError;

    fn try_from(value: RigAssistantContent) -> Result<Self, Self::Error> {
        match value.0 {
            AssistantContent::Text(text) => Ok(aws_bedrock::ContentBlock::Text(text.text)),
            AssistantContent::ToolCall(tool_call) => {
                let doc: AwsDocument = tool_call.function.arguments.into();
                Ok(aws_bedrock::ContentBlock::ToolUse(
                    aws_bedrock::ToolUseBlock::builder()
                        .tool_use_id(tool_call.id)
                        .name(tool_call.function.name)
                        .input(doc.0)
                        .build()
                        .map_err(|e| CompletionError::ProviderError(e.to_string()))?,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::assistant_content::RigAssistantContent;

    use super::AwsConverseOutput;
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use rig::{completion, message::AssistantContent, OneOrMany};

    #[test]
    fn aws_converse_output_to_completion_response() {
        let message = aws_bedrock::Message::builder()
            .role(aws_bedrock::ConversationRole::Assistant)
            .content(aws_bedrock::ContentBlock::Text("txt".into()))
            .build()
            .unwrap();
        let output = aws_bedrock::ConverseOutput::Message(message);
        let converse_output =
            aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
                .output(output)
                .stop_reason(aws_bedrock::StopReason::EndTurn)
                .build()
                .unwrap();
        let completion: Result<completion::CompletionResponse<AwsConverseOutput>, _> =
            AwsConverseOutput(converse_output).try_into();
        assert_eq!(completion.is_ok(), true);
        let completion = completion.unwrap();
        assert_eq!(
            completion.choice,
            OneOrMany::one(AssistantContent::Text("txt".into()))
        );
    }

    #[test]
    fn aws_content_block_to_assistant_content() {
        let content_block = aws_bedrock::ContentBlock::Text("text".into());
        let rig_assistant_content: Result<RigAssistantContent, _> = content_block.try_into();
        assert_eq!(rig_assistant_content.is_ok(), true);
        assert_eq!(
            rig_assistant_content.unwrap().0,
            AssistantContent::Text("text".into())
        );
    }
}
