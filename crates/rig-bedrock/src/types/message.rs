use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::NonEmpty;
use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Issuer, Message, UserContent};

use super::{
    assistant_content::RigAssistantContent,
    converse_output::{ConversationRole, Message as ConverseMessage},
    user_content::RigUserContent,
};

pub struct RigMessage(pub Message);

impl RigMessage {
    /// The Converse message, replaying the reasoning `issuer` issued.
    pub(crate) fn into_aws(self, issuer: &Issuer) -> Result<aws_bedrock::Message, ProviderError> {
        let result = match self.0 {
            Message::System { .. } => {
                return Err(ProviderError::Provider(
                    "System messages must be sent via Bedrock system blocks".to_string(),
                ));
            }
            Message::User { content } => {
                let message_content = content
                    .into_iter()
                    .map(|user_content| RigUserContent(user_content).try_into())
                    .collect::<Result<Vec<Vec<_>>, _>>()
                    .map_err(ProviderError::request)
                    .map(|nested| nested.into_iter().flatten().collect())?;

                aws_bedrock::Message::builder()
                    .role(aws_bedrock::ConversationRole::User)
                    .set_content(Some(message_content))
                    .build()
                    .map_err(ProviderError::request)?
            }
            Message::Assistant { content, .. } => aws_bedrock::Message::builder()
                .role(aws_bedrock::ConversationRole::Assistant)
                .set_content(Some(
                    // `Ok(None)` items degrade away (foreign opaque
                    // reasoning Bedrock cannot carry); errors still fail.
                    content
                        .into_iter()
                        .map(|content| RigAssistantContent(content).into_content_block(issuer))
                        .collect::<Result<Vec<Option<aws_bedrock::ContentBlock>>, _>>()?
                        .into_iter()
                        .flatten()
                        .collect(),
                ))
                .build()
                .map_err(ProviderError::request)?,
        };
        Ok(result)
    }
}

impl TryFrom<ConverseMessage> for RigMessage {
    type Error = ProviderError;

    fn try_from(message: ConverseMessage) -> Result<Self, Self::Error> {
        match message.role {
            ConversationRole::Assistant => {
                let assistant_content = message
                    .content
                    .into_iter()
                    .map(std::convert::TryInto::try_into)
                    .collect::<Result<Vec<RigAssistantContent>, _>>()?
                    .into_iter()
                    .map(|rig_assistant_content| rig_assistant_content.0)
                    .collect::<Vec<AssistantContent>>();

                let content = NonEmpty::from_vec(assistant_content).map_err(|_| {
                    ProviderError::Response(rig_core::message::EMPTY_RESPONSE_ERROR.to_owned())
                })?;

                Ok(RigMessage(Message::Assistant { content, id: None }))
            }
            ConversationRole::User => {
                let user_content = message
                    .content
                    .into_iter()
                    .map(std::convert::TryInto::try_into)
                    .collect::<Result<Vec<RigUserContent>, _>>()?
                    .into_iter()
                    .map(|user_content| user_content.0)
                    .collect::<Vec<UserContent>>();

                let content = NonEmpty::from_vec(user_content).map_err(|_| {
                    ProviderError::Response(
                        "Bedrock returned a user message with no content".to_owned(),
                    )
                })?;
                Ok(RigMessage(Message::User { content }))
            }
            _ => Err(ProviderError::Provider(
                "AWS Bedrock returned unsupported ConversationRole".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests;
