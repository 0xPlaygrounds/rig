use super::{
    ANTHROPIC_BASE_URLS, CHINA_ANTHROPIC_API_BASE_URL, CHINA_API_BASE_URL,
    GLOBAL_ANTHROPIC_API_BASE_URL, GLOBAL_API_BASE_URL,
};

#[test]
fn test_client_initialization() {
    let _client = crate::providers::minimax::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new()");
    let _client_from_builder = crate::providers::minimax::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder()");
    let _anthropic_client = crate::providers::minimax::AnthropicClient::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("AnthropicClient::new()");
    let _anthropic_client_from_builder = crate::providers::minimax::AnthropicClient::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("AnthropicClient::builder()");
}

#[test]
fn normalize_openai_bases_to_anthropic_bases() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize(GLOBAL_API_BASE_URL)
            .as_deref(),
        Some(GLOBAL_ANTHROPIC_API_BASE_URL)
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS.normalize(CHINA_API_BASE_URL).as_deref(),
        Some(CHINA_ANTHROPIC_API_BASE_URL)
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/v1")
            .as_deref(),
        Some("https://proxy.example.com/anthropic")
    );
}

#[test]
fn normalize_preserves_existing_anthropic_base() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize(CHINA_ANTHROPIC_API_BASE_URL)
            .as_deref(),
        Some(CHINA_ANTHROPIC_API_BASE_URL)
    );
}

#[test]
fn anthropic_primary_override_wins() {
    let override_url = ANTHROPIC_BASE_URLS.resolve(
        Some("https://primary.example.com/anthropic"),
        Some(CHINA_API_BASE_URL),
    );

    assert_eq!(
        override_url.as_deref(),
        Some("https://primary.example.com/anthropic")
    );
}
