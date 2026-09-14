use crate::types::message::RigMessage;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::message::{Message, UserContent};

#[test]
fn message_to_aws_message() {
    let message = Message::User {
        content: vec![UserContent::Text("text".into())],
    };
    let aws_message: Result<aws_bedrock::Message, _> = RigMessage(message).try_into();
    assert!(aws_message.is_ok());
    let aws_message = aws_message.unwrap();
    assert_eq!(aws_message.role, aws_bedrock::ConversationRole::User);
    assert_eq!(
        aws_message.content,
        vec![aws_bedrock::ContentBlock::Text("text".into())]
    );
}

/// Synthetic SDK input exercises tolerated empty IDs, not a claim about live Bedrock output.
#[test]
fn missing_call_ids_reserve_later_explicit_handles() {
    let wire = aws_bedrock::Message::builder()
        .role(aws_bedrock::ConversationRole::Assistant)
        .set_content(Some(
            (0..3)
                .map(|i| {
                    aws_bedrock::ContentBlock::ToolUse(
                        aws_bedrock::ToolUseBlock::builder()
                            .tool_use_id(if i == 1 { "tool-0" } else { "" })
                            .name("same")
                            .input(aws_smithy_types::Document::Object(Default::default()))
                            .build()
                            .unwrap(),
                    )
                })
                .collect(),
        ))
        .build()
        .unwrap();
    let convert = || {
        let message = crate::types::converse_output::Message::try_from(wire.clone()).unwrap();
        RigMessage::try_from(message).unwrap().0
    };
    let first = convert();
    assert_eq!(first, convert());
    let Message::Assistant { content, .. } = first else {
        panic!("assistant");
    };
    let calls: Vec<_> = content
        .iter()
        .filter_map(|item| match item {
            rig_core::message::AssistantContent::ToolCall(call) => Some(call),
            _ => None,
        })
        .collect();
    assert_eq!(calls.len(), 3);
    assert_eq!(
        calls
            .iter()
            .map(|call| &call.id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3
    );
    assert!(calls[0].provider.is_none());
    assert_eq!(calls[1].provider.as_ref().unwrap().call_id, "tool-0");
    assert!(calls[2].provider.is_none());
}
