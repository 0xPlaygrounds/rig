use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::Message;

use super::{assistant_content, user_content};

/// The Converse message for `message`, or `None` for an assistant turn with
/// nothing to send: Converse rejects an empty message. `signatures` is
/// whether the target model reads reasoning signatures.
pub(crate) fn to_aws(
    message: Message,
    signatures: bool,
) -> Result<Option<aws_bedrock::Message>, ProviderError> {
    let (role, content) = match message {
        Message::System { .. } => {
            return Err(ProviderError::Provider(
                "System messages must be sent via Bedrock system blocks".to_string(),
            ));
        }
        Message::User { content } => {
            let mut blocks = Vec::new();
            for part in content {
                blocks.extend(user_content::to_aws(part)?);
            }
            if blocks.is_empty() {
                blocks.push(aws_bedrock::ContentBlock::Text(
                    user_content::EMPTY_TEXT.to_owned(),
                ));
            }
            (aws_bedrock::ConversationRole::User, blocks)
        }
        Message::Assistant(turn) => {
            let mut blocks = Vec::new();
            for part in turn.content {
                blocks.extend(assistant_content::to_aws(part, signatures)?);
            }
            if blocks.is_empty() {
                return Ok(None);
            }
            (aws_bedrock::ConversationRole::Assistant, blocks)
        }
    };
    aws_bedrock::Message::builder()
        .role(role)
        .set_content(Some(content))
        .build()
        .map(Some)
        .map_err(ProviderError::request)
}

#[cfg(test)]
mod tests;
