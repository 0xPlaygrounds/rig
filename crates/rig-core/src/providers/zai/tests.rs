use super::{
    ANTHROPIC_API_BASE_URL, ANTHROPIC_BASE_URLS, CODING_API_BASE_URL, GENERAL_API_BASE_URL,
};

#[test]
fn test_client_initialization() {
    let _client = crate::providers::zai::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new()");
    let _client_from_builder = crate::providers::zai::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder()");
    let _anthropic_client = crate::providers::zai::AnthropicClient::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("AnthropicClient::new()");
    let _anthropic_client_from_builder = crate::providers::zai::AnthropicClient::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("AnthropicClient::builder()");
}

#[test]
fn normalize_openai_style_bases_to_anthropic_base() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize(GENERAL_API_BASE_URL)
            .as_deref(),
        Some(ANTHROPIC_API_BASE_URL)
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize(CODING_API_BASE_URL)
            .as_deref(),
        Some(ANTHROPIC_API_BASE_URL)
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/api/paas/v4")
            .as_deref(),
        Some("https://proxy.example.com/api/anthropic")
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/api/coding/paas/v4")
            .as_deref(),
        Some("https://proxy.example.com/api/anthropic")
    );
}

#[test]
fn normalize_preserves_existing_anthropic_base() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/api/anthropic")
            .as_deref(),
        Some("https://proxy.example.com/api/anthropic")
    );
}

#[test]
fn anthropic_primary_override_wins() {
    let override_url = ANTHROPIC_BASE_URLS.resolve(
        Some("https://primary.example.com/api/anthropic"),
        Some(GENERAL_API_BASE_URL),
    );

    assert_eq!(
        override_url.as_deref(),
        Some("https://primary.example.com/api/anthropic")
    );
}
