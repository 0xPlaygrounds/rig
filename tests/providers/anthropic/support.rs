use rig::providers::anthropic;

use crate::cassettes::ProviderCassette;

pub(super) async fn anthropic_cassette(
    scenario: &'static str,
) -> (ProviderCassette, anthropic::Client) {
    let cassette =
        ProviderCassette::start("anthropic", scenario, "https://api.anthropic.com").await;
    let client = anthropic::Client::builder()
        .api_key(cassette.api_key("ANTHROPIC_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}
