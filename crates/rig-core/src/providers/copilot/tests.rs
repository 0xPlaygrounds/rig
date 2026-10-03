use super::{CopilotIntent, base_url_from_token, default_headers};

/// The envelope declares the conversation intent, and the default is the
/// chat panel. Both routes stamp this same header set.
#[test]
fn copilot_intent_headers_use_panel_by_default_and_edits_when_requested() {
    let panel_headers = default_headers("token", "user", false, CopilotIntent::default());
    assert_eq!(
        panel_headers
            .iter()
            .find(|(name, _)| *name == "openai-intent")
            .map(|(_, value)| value.as_str()),
        Some("conversation-panel")
    );

    let edits_headers = default_headers("token", "user", false, CopilotIntent::Edits);
    assert_eq!(
        edits_headers
            .iter()
            .find(|(name, _)| *name == "openai-intent")
            .map(|(_, value)| value.as_str()),
        Some("conversation-edits")
    );
}

/// A vision request is gated behind a header the other requests must not
/// carry.
#[test]
fn only_a_vision_request_carries_the_vision_header() {
    let vision = |has_vision| {
        default_headers("token", "user", has_vision, CopilotIntent::default())
            .iter()
            .any(|(name, value)| *name == "copilot-vision-request" && value == "true")
    };
    assert!(vision(true));
    assert!(!vision(false));
}

#[test]
fn base_url_from_token_derives_api_endpoint() {
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=proxy.individual.githubcopilot.com;exp=2").as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com;exp=2")
            .as_deref(),
        Some("https://api.individual.githubcopilot.com")
    );
    assert_eq!(base_url_from_token("tid=1;exp=2"), None);
}

#[test]
fn base_url_from_token_rejects_unsafe_or_non_copilot_endpoints() {
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=http://proxy.individual.githubcopilot.com;exp=2"),
        None
    );
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://evil.example.com;exp=2"),
        None
    );
    assert_eq!(base_url_from_token("tid=1;proxy-ep=://bad;exp=2"), None);
    assert_eq!(base_url_from_token("tid=1;proxy-ep=;exp=2"), None);
    assert_eq!(
        base_url_from_token("tid=1;proxy-ep=https://proxy.individual.githubcopilot.com/base;exp=2"),
        None
    );
}
