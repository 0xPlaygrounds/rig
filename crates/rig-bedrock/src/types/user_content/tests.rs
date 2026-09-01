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
    assert_eq!(content.unwrap().text, "42");
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
    );
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
    );
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
