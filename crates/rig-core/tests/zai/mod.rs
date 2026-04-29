mod anthropic;
mod coding;
mod general;

use rig_core::providers::zai;

pub(crate) fn api_key() -> String {
    std::env::var("ZAI_API_KEY").expect("ZAI_API_KEY should be set")
}

pub(crate) fn general_client() -> zai::Client {
    zai::Client::builder()
        .api_key(api_key())
        .general()
        .build()
        .expect("Z.AI general client should build")
}

pub(crate) fn coding_client() -> zai::Client {
    zai::Client::builder()
        .api_key(api_key())
        .coding()
        .build()
        .expect("Z.AI coding client should build")
}

pub(crate) fn anthropic_client() -> zai::AnthropicClient {
    zai::AnthropicClient::builder()
        .api_key(api_key())
        .general()
        .build()
        .expect("Z.AI Anthropic-compatible client should build")
}
