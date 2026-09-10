use super::AnthropicBaseUrl;

const RULE: AnthropicBaseUrl = AnthropicBaseUrl::new(
    &[(
        "https://api.example.com/v1",
        "https://api.example.com/anthropic",
    )],
    &["/v1", "/v1/"],
    "/anthropic",
);

#[test]
fn maps_known_and_custom_openai_bases() {
    assert_eq!(
        RULE.normalize("https://api.example.com/v1/").as_deref(),
        Some("https://api.example.com/anthropic")
    );
    assert_eq!(
        RULE.normalize("https://proxy.example.com/v1").as_deref(),
        Some("https://proxy.example.com/anthropic")
    );
}

#[test]
fn primary_wins_and_unknown_fallback_paths_are_ignored() {
    assert_eq!(
        RULE.resolve(
            Some("https://primary.example.com/anthropic"),
            Some("https://proxy.example.com/v1")
        )
        .as_deref(),
        Some("https://primary.example.com/anthropic")
    );
    assert_eq!(
        RULE.resolve(None, Some("https://proxy.example.com/api")),
        None
    );
}
