use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{AssistantContent, Text},
};
use serde::{Deserialize, Serialize};

use crate::types::message::RigMessage;

use super::{
    converse_output::{ContentBlock, InternalConverseOutput, TokenUsage},
    json::AwsDocument,
};
use rig::completion::{self, GetTokenUsage};
use rig::telemetry::ProviderResponseExt;

#[derive(Clone, Deserialize, Serialize)]
pub struct AwsConverseOutput(pub InternalConverseOutput);

fn normalize_usage(usage: &TokenUsage) -> completion::Usage {
    completion::Usage {
        input_tokens: usage.input_tokens as u64,
        output_tokens: usage.output_tokens as u64,
        total_tokens: usage.total_tokens as u64,
        cached_input_tokens: usage.cache_read_input_tokens.unwrap_or_default() as u64,
        cache_creation_input_tokens: usage.cache_write_input_tokens.unwrap_or_default() as u64,
    }
}

impl ProviderResponseExt for AwsConverseOutput {
    type OutputMessage = serde_json::Value;
    type Usage = completion::Usage;

    fn get_response_id(&self) -> Option<String> {
        None // Bedrock Converse API doesn't return a response ID
    }

    fn get_response_model_name(&self) -> Option<String> {
        None // Bedrock doesn't echo model name in response
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.0
            .output
            .as_ref()
            .map(|output| vec![serde_json::to_value(output).unwrap_or_default()])
            .unwrap_or_default()
    }

    fn get_text_response(&self) -> Option<String> {
        let output = self.0.output.as_ref()?;
        let message = output.as_message().ok()?;
        let response = message
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if response.is_empty() {
            None
        } else {
            Some(response)
        }
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.0.usage().map(normalize_usage)
    }
}

impl GetTokenUsage for AwsConverseOutput {
    fn token_usage(&self) -> Option<completion::Usage> {
        self.get_usage()
    }
}

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

        let usage = value.0.usage().map(normalize_usage).unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: value,
            message_id: None,
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
                        rig::message::Reasoning::new_with_signature(
                            &reasoning_text.text,
                            reasoning_text.signature,
                        ),
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
                let signed_text_count = reasoning
                    .content
                    .iter()
                    .filter(|content| {
                        matches!(
                            content,
                            rig::message::ReasoningContent::Text {
                                signature: Some(_),
                                ..
                            }
                        )
                    })
                    .count();
                if signed_text_count > 1 {
                    return Err(CompletionError::ProviderError(
                        "AWS Bedrock does not support multiple signed reasoning text blocks"
                            .to_owned(),
                    ));
                }
                if signed_text_count == 1 && reasoning.content.len() > 1 {
                    return Err(CompletionError::ProviderError(
                        "AWS Bedrock requires a single signed reasoning text block without additional reasoning parts"
                            .to_owned(),
                    ));
                }

                let flattened_text = reasoning.display_text();
                let has_signature = reasoning.first_signature().is_some();
                // Adaptive thinking on Bedrock can emit a reasoning block whose
                // plaintext body is empty but with a real cryptographic
                // signature attached. The signature is what Anthropic uses to
                // verify tool_use round-trips, so we must preserve it. Only
                // reject when there's neither text nor signature to send.
                if flattened_text.is_empty() && !has_signature {
                    return Err(CompletionError::ProviderError(
                        "AWS Bedrock reasoning conversion requires at least one text or summary block"
                            .to_owned(),
                    ));
                }

                let mut reasoning_block =
                    aws_bedrock::ReasoningTextBlock::builder().text(flattened_text);

                if let Some(sig) = reasoning.first_signature().map(str::to_owned) {
                    reasoning_block = reasoning_block.signature(sig);
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
            AssistantContent::Image(_) => Err(CompletionError::ProviderError(
                "AWS Bedrock does not support image content in assistant messages".to_owned(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::{
        assistant_content::RigAssistantContent, converse_output::InternalConverseOutput,
        errors::TypeConversionError, json::AwsDocument,
    };

    use super::AwsConverseOutput;
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use rig::{
        OneOrMany, completion,
        completion::GetTokenUsage,
        message::{AssistantContent, ReasoningContent},
        telemetry::ProviderResponseExt,
    };
    use serde_json::json;

    /// Helper: build an AwsConverseOutput with text content and optional usage.
    fn make_output(text: &str, usage: Option<aws_bedrock::TokenUsage>) -> AwsConverseOutput {
        make_output_with_content(vec![aws_bedrock::ContentBlock::Text(text.into())], usage)
    }

    fn make_output_with_content(
        content: Vec<aws_bedrock::ContentBlock>,
        usage: Option<aws_bedrock::TokenUsage>,
    ) -> AwsConverseOutput {
        let message = aws_bedrock::Message::builder()
            .role(aws_bedrock::ConversationRole::Assistant)
            .set_content(Some(content))
            .build()
            .unwrap();
        let mut builder = aws_sdk_bedrockruntime::operation::converse::ConverseOutput::builder()
            .output(aws_bedrock::ConverseOutput::Message(message))
            .stop_reason(aws_bedrock::StopReason::EndTurn);
        if let Some(u) = usage {
            builder = builder.usage(u);
        }
        let internal: InternalConverseOutput = builder.build().unwrap().try_into().unwrap();
        AwsConverseOutput(internal)
    }

    fn make_usage(input: i32, output: i32, total: i32) -> aws_bedrock::TokenUsage {
        aws_bedrock::TokenUsage::builder()
            .input_tokens(input)
            .output_tokens(output)
            .total_tokens(total)
            .build()
            .unwrap()
    }

    #[test]
    fn provider_response_ext_get_text_response() {
        let out = make_output("hello world", None);
        assert_eq!(out.get_text_response(), Some("hello world".to_string()));
    }

    #[test]
    fn provider_response_ext_response_id_is_none() {
        let out = make_output("x", None);
        assert!(out.get_response_id().is_none());
        assert!(out.get_response_model_name().is_none());
    }

    #[test]
    fn provider_response_ext_get_usage_with_tokens() {
        let out = make_output("x", Some(make_usage(100, 50, 150)));
        let usage = out.get_usage().unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn provider_response_ext_get_usage_none_when_missing() {
        let out = make_output("x", None);
        assert!(out.get_usage().is_none());
    }

    #[test]
    fn provider_response_ext_output_messages_serializable() {
        let out = make_output("test", None);
        let msgs = out.get_output_messages();
        assert_eq!(msgs.len(), 1);
        // Should be valid JSON
        assert!(msgs[0].is_object());
    }

    #[test]
    fn get_token_usage_delegates_to_provider_response_ext() {
        let out = make_output("x", Some(make_usage(10, 20, 30)));
        let usage = out.token_usage().unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }

    #[test]
    fn get_token_usage_none_when_no_usage() {
        let out = make_output("x", None);
        assert!(out.token_usage().is_none());
    }

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
    fn aws_converse_output_preserves_parallel_tool_calls_in_completion_response() {
        let content = vec![
            aws_bedrock::ContentBlock::Text("preface".into()),
            aws_bedrock::ContentBlock::ToolUse(
                aws_bedrock::ToolUseBlock::builder()
                    .tool_use_id("call_1")
                    .name("add")
                    .input(AwsDocument::from(json!({"x": 1, "y": 2})).0)
                    .build()
                    .unwrap(),
            ),
            aws_bedrock::ContentBlock::ToolUse(
                aws_bedrock::ToolUseBlock::builder()
                    .tool_use_id("call_2")
                    .name("subtract")
                    .input(AwsDocument::from(json!({"x": 4, "y": 3})).0)
                    .build()
                    .unwrap(),
            ),
        ];

        let completion: completion::CompletionResponse<AwsConverseOutput> =
            make_output_with_content(content, None)
                .try_into()
                .expect("conversion should succeed");

        let choice: Vec<_> = completion.choice.into_iter().collect();
        assert_eq!(choice.len(), 3);
        assert_eq!(choice[0], AssistantContent::Text("preface".into()));

        let AssistantContent::ToolCall(first_tool) = &choice[1] else {
            panic!("expected first tool call");
        };
        assert_eq!(first_tool.id, "call_1");
        assert_eq!(first_tool.function.name, "add");
        assert_eq!(first_tool.function.arguments, json!({"x": 1, "y": 2}));

        let AssistantContent::ToolCall(second_tool) = &choice[2] else {
            panic!("expected second tool call");
        };
        assert_eq!(second_tool.id, "call_2");
        assert_eq!(second_tool.function.name, "subtract");
        assert_eq!(second_tool.function.arguments, json!({"x": 4, "y": 3}));
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
                assert_eq!(reasoning.first_text(), Some("This is my reasoning"));
                assert_eq!(reasoning.first_signature(), None);
                assert!(matches!(
                    reasoning.content.first(),
                    Some(ReasoningContent::Text { text, signature: None }) if text == "This is my reasoning"
                ));
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
                    reasoning.first_text(),
                    Some("This is my reasoning with signature")
                );
                assert_eq!(reasoning.first_signature(), Some("test_signature_123"));
                assert!(matches!(
                    reasoning.content.first(),
                    Some(ReasoningContent::Text { text, signature: Some(sig) })
                        if text == "This is my reasoning with signature" && sig == "test_signature_123"
                ));
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
        let reasoning = rig::message::Reasoning::new_with_signature(
            "My reasoning content",
            Some("sig_abc_123".to_string()),
        );
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
        let reasoning = rig::message::Reasoning::multi(vec![
            "First part".to_string(),
            " Second part".to_string(),
            " Third part".to_string(),
        ]);

        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(aws_content_block.is_ok());

        match aws_content_block.unwrap() {
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
            ) => {
                assert_eq!(reasoning_text.text, "First part\n Second part\n Third part");
            }
            _ => panic!("Expected ContentBlock::ReasoningContent"),
        }
    }

    #[test]
    fn rig_reasoning_with_empty_text_and_signature_is_converted() {
        // Adaptive thinking on Bedrock can emit a reasoning block whose
        // plaintext body is empty but with a real cryptographic signature
        // attached. Verify we forward this as a `ReasoningTextBlock` with
        // empty text + signature instead of rejecting it.
        let reasoning =
            rig::message::Reasoning::new_with_signature("", Some("sig_empty_text".to_string()));
        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(aws_content_block.is_ok());

        match aws_content_block.unwrap() {
            aws_bedrock::ContentBlock::ReasoningContent(
                aws_bedrock::ReasoningContentBlock::ReasoningText(reasoning_text),
            ) => {
                assert_eq!(reasoning_text.text, "");
                assert_eq!(reasoning_text.signature, Some("sig_empty_text".to_string()));
            }
            _ => panic!("Expected ContentBlock::ReasoningContent"),
        }
    }

    #[test]
    fn rig_reasoning_with_empty_text_and_no_signature_returns_error() {
        let reasoning = rig::message::Reasoning::new_with_signature("", None);
        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(matches!(
            aws_content_block,
            Err(completion::CompletionError::ProviderError(message))
                if message.contains("at least one text or summary block")
        ));
    }

    #[test]
    fn rig_reasoning_with_multiple_signed_text_blocks_returns_error() {
        let mut reasoning =
            rig::message::Reasoning::new_with_signature("part one", Some("sig_1".to_string()));
        reasoning.content.push(ReasoningContent::Text {
            text: "part two".to_string(),
            signature: Some("sig_2".to_string()),
        });
        let rig_content = RigAssistantContent(AssistantContent::Reasoning(reasoning));

        let aws_content_block: Result<aws_bedrock::ContentBlock, _> = rig_content.try_into();
        assert!(matches!(
            aws_content_block,
            Err(completion::CompletionError::ProviderError(message))
                if message.contains("multiple signed reasoning text blocks")
        ));
    }
}
