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
