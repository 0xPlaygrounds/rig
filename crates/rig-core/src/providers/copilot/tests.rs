use super::auth::base_url_from_token;

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
