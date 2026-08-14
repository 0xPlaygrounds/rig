use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::{
    completion::CompletionError,
    message::{AssistantContent, Message, UserContent},
};

use super::{
    assistant_content::RigAssistantContent,
    converse_output::{ConversationRole, Message as ConverseMessage},
    user_content::RigUserContent,
};

pub struct RigMessage(pub Message);

impl TryFrom<RigMessage> for aws_bedrock::Message {
    type Error = CompletionError;

    fn try_from(value: RigMessage) -> Result<Self, Self::Error> {
        let result = match value.0 {
            Message::System { .. } => {
                return Err(CompletionError::ProviderError(
                    "System messages must be sent via Bedrock system blocks".to_string(),
                ));
            }
            Message::User { content } => {
                let message_content = content
                    .into_iter()
                    .map(|user_content| RigUserContent(user_content).try_into())
                    .collect::<Result<Vec<Vec<_>>, _>>()
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))
                    .map(|nested| nested.into_iter().flatten().collect())?;

                aws_bedrock::Message::builder()
                    .role(aws_bedrock::ConversationRole::User)
                    .set_content(Some(message_content))
                    .build()
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?
            }
            Message::Assistant { content, .. } => aws_bedrock::Message::builder()
                .role(aws_bedrock::ConversationRole::Assistant)
                .set_content(Some(
                    // `Ok(None)` items degrade away (foreign opaque
                    // reasoning Bedrock cannot carry); errors still fail.
                    content
                        .into_iter()
                        .map(|content| RigAssistantContent(content).into_content_block())
                        .collect::<Result<Vec<Option<aws_bedrock::ContentBlock>>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect(),
                ))
                .build()
                .map_err(|e| CompletionError::RequestError(Box::new(e)))?,
        };
        Ok(result)
    }
}

impl TryFrom<ConverseMessage> for RigMessage {
    type Error = CompletionError;

    fn try_from(message: ConverseMessage) -> Result<Self, Self::Error> {
        match message.role {
            ConversationRole::Assistant => {
                let assistant_content = message
                    .content
                    .into_iter()
                    .map(|c| c.try_into())
                    .collect::<Result<Vec<RigAssistantContent>, _>>()?
                    .into_iter()
                    .map(|rig_assistant_content| rig_assistant_content.0)
                    .collect::<Vec<AssistantContent>>();

                let content = rig_core::message::require_non_empty_response(assistant_content)?;

                Ok(RigMessage(Message::Assistant { content, id: None }))
            }
            ConversationRole::User => {
                let user_content = message
                    .content
                    .into_iter()
                    .map(|c| c.try_into())
                    .collect::<Result<Vec<RigUserContent>, _>>()?
                    .into_iter()
                    .map(|user_content| user_content.0)
                    .collect::<Vec<UserContent>>();

                let content = rig_core::message::require_non_empty(user_content, || {
                    CompletionError::ResponseError(
                        "Bedrock returned a user message with no content".to_owned(),
                    )
                })?;
                Ok(RigMessage(Message::User { content }))
            }
            _ => Err(CompletionError::ProviderError(
                "AWS Bedrock returned unsupported ConversationRole".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
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
}
