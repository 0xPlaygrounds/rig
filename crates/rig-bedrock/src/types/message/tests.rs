use crate::completion::Family;
use crate::types::message;
use aws_sdk_bedrockruntime::types as aws_bedrock;
use rig_core::message::{AssistantContent, Message, UserContent};

#[test]
fn message_to_aws_message() {
    let rig_message = Message::User {
        content: vec![UserContent::Text("text".into())],
    };
    let aws_message = message::to_aws(rig_message, Family::Other)
        .unwrap()
        .unwrap();
    assert_eq!(aws_message.role, aws_bedrock::ConversationRole::User);
    assert_eq!(
        aws_message.content,
        vec![aws_bedrock::ContentBlock::Text("text".into())]
    );
}

/// Converse rejects blank text and empty messages: blank text is never
/// sent, an assistant turn left with nothing is left out, and a user
/// message left with nothing sends pi's placeholder.
#[test]
fn blank_text_is_never_sent_and_no_message_is_empty() {
    let assistant = Message::from(vec![
        AssistantContent::text(" "),
        AssistantContent::reasoning(""),
    ]);
    assert!(
        message::to_aws(assistant, Family::Claude)
            .unwrap()
            .is_none()
    );

    let user = Message::User {
        content: vec![UserContent::text("\n")],
    };
    let aws_message = message::to_aws(user, Family::Claude).unwrap().unwrap();
    assert_eq!(
        aws_message.content,
        vec![aws_bedrock::ContentBlock::Text("<empty>".into())]
    );

    let kept = Message::from(vec![
        AssistantContent::text(""),
        AssistantContent::text("answer"),
    ]);
    let aws_message = message::to_aws(kept, Family::Claude).unwrap().unwrap();
    assert_eq!(
        aws_message.content,
        vec![aws_bedrock::ContentBlock::Text("answer".into())]
    );
}

/// A blank system message has nothing to send.
#[test]
fn a_blank_system_message_is_not_sent() {
    assert!(
        message::to_aws(Message::system(" "), Family::Other)
            .unwrap()
            .is_none()
    );
}
