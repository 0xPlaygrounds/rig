mod auth;
mod completion;
mod streaming;

use rig::message::AssistantContent;
use rig::providers::chatgpt::{self, ChatGPTAuth};

pub(crate) fn live_builder() -> chatgpt::ClientBuilder {
    let mut builder = chatgpt::Client::builder();

    if let Ok(base_url) =
        std::env::var("CHATGPT_API_BASE").or_else(|_| std::env::var("OPENAI_CHATGPT_API_BASE"))
    {
        builder = builder.base_url(base_url);
    }

    if let Ok(access_token) = std::env::var("CHATGPT_ACCESS_TOKEN") {
        let account_id = std::env::var("CHATGPT_ACCOUNT_ID").ok();
        builder.api_key(ChatGPTAuth::AccessToken {
            access_token,
            account_id,
        })
    } else {
        builder.oauth()
    }
}

pub(crate) fn live_client() -> chatgpt::Client {
    live_builder().build().expect("ChatGPT client should build")
}

pub(crate) fn response_text(choice: &rig::OneOrMany<AssistantContent>) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}
