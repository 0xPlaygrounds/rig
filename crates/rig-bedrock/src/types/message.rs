use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::{AssistantContent, Issuer, Message};

use super::{
    assistant_content,
    converse_output::{ConversationRole, Message as ConverseMessage},
    user_content,
};

/// The Converse message for `message`, replaying the reasoning `issuer`
/// issued.
pub(crate) fn to_aws(
    message: Message,
    issuer: &Issuer,
) -> Result<aws_bedrock::Message, ProviderError> {
    let result = match message {
        Message::System { .. } => {
            return Err(ProviderError::Provider(
                "System messages must be sent via Bedrock system blocks".to_string(),
            ));
        }
        Message::User { content } => {
            let message_content = content
                .into_iter()
                .map(user_content::to_aws)
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
                    .map(|content| assistant_content::to_aws(content, issuer))
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

/// The content of a Converse reply message. A reply is the assistant's turn,
/// so a reply in any other role is an error.
pub(crate) fn assistant_reply(
    message: ConverseMessage,
) -> Result<Vec<AssistantContent>, ProviderError> {
    match message.role {
        ConversationRole::Assistant => {
            let content = message
                .content
                .into_iter()
                .map(assistant_content::from_converse)
                .collect::<Result<Vec<_>, _>>()?;
            rig_core::message::require_non_empty_response(content)
        }
        ConversationRole::User => Err(ProviderError::Response(
            "Converse output message was not an assistant message".to_owned(),
        )),
        ConversationRole::Unknown(_) => Err(ProviderError::Provider(
            "AWS Bedrock returned unsupported ConversationRole".into(),
        )),
    }
}

#[cfg(test)]
mod tests;
