use super::{ANTHROPIC_API_BASE_URL, ANTHROPIC_BASE_URLS, API_BASE_URL};

#[test]
fn test_client_initialization() {
    let _client = crate::providers::xiaomimimo::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new()");
    let _client_from_builder = crate::providers::xiaomimimo::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder()");
    let _anthropic_client = crate::providers::xiaomimimo::AnthropicClient::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("AnthropicClient::new()");
    let _anthropic_client_from_builder = crate::providers::xiaomimimo::AnthropicClient::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("AnthropicClient::builder()");
}

#[test]
fn normalize_openai_bases_to_anthropic_bases() {
    assert_eq!(
        ANTHROPIC_BASE_URLS.normalize(API_BASE_URL).as_deref(),
        Some(ANTHROPIC_API_BASE_URL)
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/v1")
            .as_deref(),
        Some("https://proxy.example.com/anthropic/v1")
    );
}

#[test]
fn normalize_preserves_existing_anthropic_base() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize(ANTHROPIC_API_BASE_URL)
            .as_deref(),
        Some(ANTHROPIC_API_BASE_URL)
    );
}

#[test]
fn anthropic_primary_override_wins() {
    let override_url = ANTHROPIC_BASE_URLS.resolve(
        Some("https://primary.example.com/anthropic/v1"),
        Some(API_BASE_URL),
    );

    assert_eq!(
        override_url.as_deref(),
        Some("https://primary.example.com/anthropic/v1")
    );
}
