use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig::{
    completion::CompletionError,
    message::{AssistantContent, Message, UserContent},
    OneOrMany,
};

use super::{
    assistant_content::RigAssistantContent,
    user_content::{RigUserContent, UserContentWithPrompt},
};

pub struct MessageWithPrompt {
    pub message: Message,
    pub prompt: Option<String>,
}

impl TryFrom<MessageWithPrompt> for aws_bedrock::Message {
    type Error = CompletionError;

    fn try_from(value: MessageWithPrompt) -> Result<Self, Self::Error> {
        let result = match value.message {
            Message::User { content } => {
                let message_content = content
                    .into_iter()
                    .map(|user_content| {
                        UserContentWithPrompt {
                            user_content,
                            prompt: value.prompt.clone(),
                        }
                        .try_into()
                    })
                    .collect::<Result<Vec<Vec<_>>, _>>()
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))
                    .map(|nested| nested.into_iter().flatten().collect())?;

                aws_bedrock::Message::builder()
                    .role(aws_bedrock::ConversationRole::User)
                    .set_content(Some(message_content))
                    .build()
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?
            }
            Message::Assistant { content } => aws_bedrock::Message::builder()
                .role(aws_bedrock::ConversationRole::Assistant)
                .set_content(Some(
                    content
                        .into_iter()
                        .map(|content| RigAssistantContent(content).try_into())
                        .collect::<Result<Vec<aws_bedrock::ContentBlock>, _>>()?,
                ))
                .build()
                .map_err(|e| CompletionError::RequestError(Box::new(e)))?,
        };
        Ok(result)
    }
}

pub struct RigMessage(pub Message);

impl TryFrom<aws_bedrock::Message> for RigMessage {
    type Error = CompletionError;

    fn try_from(message: aws_bedrock::Message) -> Result<Self, Self::Error> {
        match message.role {
            aws_bedrock::ConversationRole::Assistant => {
                let assistant_content = message
                    .content
                    .into_iter()
                    .map(|c| c.try_into())
                    .collect::<Result<Vec<RigAssistantContent>, _>>()?
                    .into_iter()
                    .map(|rig_assistant_content| rig_assistant_content.0)
                    .collect::<Vec<AssistantContent>>();

                let content = OneOrMany::many(assistant_content)
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                Ok(RigMessage(Message::Assistant { content }))
            }
            aws_bedrock::ConversationRole::User => {
                let user_content = message
                    .content
                    .into_iter()
                    .map(|c| c.try_into())
                    .collect::<Result<Vec<RigUserContent>, _>>()?
                    .into_iter()
                    .map(|user_content| user_content.0)
                    .collect::<Vec<UserContent>>();

                let content = OneOrMany::many(user_content)
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;
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
    use aws_sdk_bedrockruntime::types as aws_bedrock;
    use rig::{
        message::{Message, UserContent},
        OneOrMany,
    };

    use super::MessageWithPrompt;

    #[test]
    fn message_with_prompt_to_aws_message() {
        let message = Message::User {
            content: OneOrMany::one(UserContent::Text("text".into())),
        };
        let message_with_prompt = MessageWithPrompt {
            prompt: Some("text".into()),
            message,
        };
        let aws_message: Result<aws_bedrock::Message, _> = message_with_prompt.try_into();
        assert_eq!(aws_message.is_ok(), true);
        let aws_message = aws_message.unwrap();
        assert_eq!(aws_message.role, aws_bedrock::ConversationRole::User);
        assert_eq!(
            aws_message.content,
            vec![aws_bedrock::ContentBlock::Text("text".into())]
        );
    }
}
