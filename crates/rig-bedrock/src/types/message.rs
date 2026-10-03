use aws_sdk_bedrockruntime::types as aws_bedrock;

use rig_core::error::ProviderError;
use rig_core::message::Message;

use super::{assistant_content, user_content};
use crate::completion::Family;

/// The Converse message for `message`, or `None` for an assistant turn with
/// nothing to send: Converse rejects an empty message. `family` is the
/// target model's. A system message here is not one of the history's
/// leading ones, so it is user text where the history puts it.
pub(crate) fn to_aws(
    message: Message,
    family: Family,
) -> Result<Option<aws_bedrock::Message>, ProviderError> {
    let (role, content) = match message {
        Message::System { content } if content.trim().is_empty() => return Ok(None),
        Message::System { content } => (
            aws_bedrock::ConversationRole::User,
            vec![aws_bedrock::ContentBlock::Text(content)],
        ),
        Message::User { content } => {
            let mut blocks = Vec::new();
            for part in content {
                blocks.extend(user_content::to_aws(part, family)?);
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
                blocks.extend(assistant_content::to_aws(part, family)?);
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
