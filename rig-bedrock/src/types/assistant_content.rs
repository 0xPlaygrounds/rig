use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    OneOrMany,
    completion::CompletionError,
    message::{AssistantContent, Text, ToolCall, ToolFunction},
};
use serde::{Deserialize, Serialize};

use crate::types::message::RigMessage;

use super::{converse_output::InternalConverseOutput, json::AwsDocument};
use rig::completion;

#[derive(Clone, Deserialize, Serialize)]
pub struct AwsConverseOutput(pub InternalConverseOutput);

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
            completion::Message::Assistant { content, .. } => Ok(content),
            _ => Err(CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )),
        }?;

        let usage = value
            .0
            .usage()
            .map(|usage| completion::Usage {
                input_tokens: usage.input_tokens as u64,
                output_tokens: usage.output_tokens as u64,
                total_tokens: usage.total_tokens as u64,
            })
            .unwrap_or_default();

        if let Some(tool_use) = choice.iter().find_map(|content| match content {
            AssistantContent::ToolCall(tool_call) => Some(tool_call.to_owned()),
            _ => None,
        }) {
            return Ok(completion::CompletionResponse {
                choice: OneOrMany::one(AssistantContent::ToolCall(ToolCall {
                    id: tool_use.id,
                    call_id: None,
                    function: ToolFunction {
                        name: tool_use.function.name,
                        arguments: tool_use.function.arguments,
                    },
                })),
                usage,
                raw_response: value,
            });
        }

        Ok(completion::CompletionResponse {
            choice,
            usage,
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
            aws_bedrock::ContentBlock::ReasoningContent(reasoning_block) => match reasoning_block {
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text) => {
                    Ok(RigAssistantContent(AssistantContent::Reasoning(
                        rig::message::Reasoning::new(&reasoning_text.text)
                            .with_signature(reasoning_text.signature),
                    )))
                }
                _ => Err(CompletionError::ProviderError(
                    "AWS Bedrock returned unsupported ReasoningContentBlock variant".into(),
                )),
            },
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
            AssistantContent::Reasoning(reasoning) => {
                let mut reasoning_block =
                    aws_bedrock::ReasoningTextBlock::builder().text(reasoning.reasoning.join(""));

                if let Some(sig) = &reasoning.signature {
                    reasoning_block = reasoning_block.signature(sig.clone());
                }

                let reasoning_text_block = reasoning_block.build().map_err(|e| {
                    CompletionError::ProviderError(format!(
                        "Failed to build reasoning block: {}",
                        e
                    ))
                })?;

                Ok(aws_bedrock::ContentBlock::ReasoningContent(
                    aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{
        assistant_content::RigAssistantContent, converse_output::InternalConverseOutput,
        errors::TypeConversionError,
    };

    use super::AwsConverseOutput;
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use rig::{OneOrMany, completion, message::AssistantContent};

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
        let converse_output: Result<InternalConverseOutput, TypeConversionError> =
            converse_output.try_into();
        assert!(converse_output.is_ok());
        let converse_output = converse_output.unwrap();
        let completion: Result<completion::CompletionResponse<AwsConverseOutput>, _> =
            AwsConverseOutput(converse_output).try_into();
        assert!(completion.is_ok());
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
        assert!(rig_assistant_content.is_ok());
        assert_eq!(
            rig_assistant_content.unwrap().0,
            AssistantContent::Text("text".into())
        );
    }

    #[test]
    fn aws_reasoning_content_to_assistant_content_without_signature() {
        // Test conversion from AWS ReasoningContent to Rig AssistantContent without signature
        let reasoning_text_block = aws_bedrock::ReasoningTextBlock::builder()
            .text("This is my reasoning")
            .build()
            .unwrap();

        let content_block = aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
        );

        let rig_assistant_content: Result<RigAssistantContent, _> = content_block.try_into();
        assert!(rig_assistant_content.is_ok());

        match rig_assistant_content.unwrap().0 {
            AssistantContent::Reasoning(reasoning) => {
                assert_eq!(reasoning.reasoning, vec!["This is my reasoning"]);
                assert_eq!(reasoning.signature, None);
            }
            _ => panic!("Expected AssistantContent::Reasoning"),
        }
    }

    #[test]
    fn aws_reasoning_content_to_assistant_content_with_signature() {
        // Test conversion from AWS ReasoningContent to Rig AssistantContent with signature
        let reasoning_text_block = aws_bedrock::ReasoningTextBlock::builder()
            .text("This is my reasoning with signature")
            .signature("test_signature_123")
            .build()
            .unwrap();

        let content_block = aws_bedrock::ContentBlock::ReasoningContent(
            aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text_block),
        );

        let rig_assistant_content: Result<RigAssistantContent, _> = content_block.try_into();
        assert!(rig_assistant_content.is_ok());

        match rig_assistant_content.unwrap().0 {
            AssistantContent::Reasoning(reasoning) => {
                assert_eq!(
                    reasoning.reasoning,
                    vec!["This is my reasoning with signature"]
                );
                assert_eq!(reasoning.signature, Some("test_signature_123".to_string()));
            }
            _ => panic!("Expected AssistantContent::Reasoning"),
        }
    }

    #[test]
    fn rig_reasoning_to_aws_content_block_without_signature() {
        // Test conversion from Rig Reasoning to AWS ContentBlock without signature
        let reasoning = rig::message::Reasoning::new("My reasoning content");
        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(aws_content_block.is_ok());

        match aws_content_block.unwrap() {
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
            ) => {
                assert_eq!(reasoning_text.text, "My reasoning content");
                assert_eq!(reasoning_text.signature, None);
            }
            _ => panic!("Expected ContentBlock::ReasoningContent"),
        }
    }

    #[test]
    fn rig_reasoning_to_aws_content_block_with_signature() {
        // Test conversion from Rig Reasoning to AWS ContentBlock with signature
        let reasoning = rig::message::Reasoning::new("My reasoning content")
            .with_signature(Some("sig_abc_123".to_string()));
        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(aws_content_block.is_ok());

        match aws_content_block.unwrap() {
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
            ) => {
                assert_eq!(reasoning_text.text, "My reasoning content");
                assert_eq!(reasoning_text.signature, Some("sig_abc_123".to_string()));
            }
            _ => panic!("Expected ContentBlock::ReasoningContent"),
        }
    }

    #[test]
    fn rig_reasoning_with_multiple_strings_to_aws_content_block() {
        // Test that multiple reasoning strings are joined correctly
        let mut reasoning = rig::message::Reasoning::new("First part");
        reasoning.reasoning.push(" Second part".to_string());
        reasoning.reasoning.push(" Third part".to_string());

        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(aws_content_block.is_ok());

        match aws_content_block.unwrap() {
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
            ) => {
                assert_eq!(reasoning_text.text, "First part Second part Third part");
            }
            _ => panic!("Expected ContentBlock::ReasoningContent"),
        }
    }
}
