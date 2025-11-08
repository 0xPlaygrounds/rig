use google_cloud_aiplatform_v1 as vertexai;
use rig::completion::CompletionError;
use rig::message::{AssistantContent, Message, Text, UserContent};

pub struct RigMessage(pub Message);

impl TryFrom<RigMessage> for vertexai::model::Content {
    type Error = CompletionError;

    fn try_from(value: RigMessage) -> Result<Self, Self::Error> {
        match value.0 {
            Message::User { content } => {
                let parts: Result<Vec<vertexai::model::Part>, _> = content
                    .into_iter()
                    .map(|user_content| match user_content {
                        UserContent::Text(Text { text }) => {
                            Ok(vertexai::model::Part::new().set_text(text))
                        }
                        _ => Err(CompletionError::ProviderError(
                            "Only text user content is supported in this initial implementation".to_string(),
                        )),
                    })
                    .collect();

                let parts = parts?;
                Ok(vertexai::model::Content::new()
                    .set_role("user")
                    .set_parts(parts))
            }
            Message::Assistant { content, .. } => {
                let parts: Result<Vec<vertexai::model::Part>, _> = content
                    .into_iter()
                    .map(|assistant_content| match assistant_content {
                        AssistantContent::Text(Text { text }) => {
                            Ok(vertexai::model::Part::new().set_text(text))
                        }
                        _ => Err(CompletionError::ProviderError(
                            "Only text assistant content is supported in this initial implementation".to_string(),
                        )),
                    })
                    .collect();

                let parts = parts?;
                Ok(vertexai::model::Content::new()
                    .set_role("model")
                    .set_parts(parts))
            }
        }
    }
}

