use rig::providers::gemini;

use crate::cassettes::ProviderCassette;

pub(super) async fn gemini_cassette(scenario: &'static str) -> (ProviderCassette, gemini::Client) {
    let cassette = ProviderCassette::start(
        "gemini",
        scenario,
        "https://generativelanguage.googleapis.com",
    )
    .await;
    let client = gemini::Client::builder()
        .api_key(cassette.api_key("GEMINI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn gemini_interactions_cassette(
    scenario: &'static str,
) -> (ProviderCassette, gemini::InteractionsClient) {
    let (cassette, client) = gemini_cassette(scenario).await;
    (cassette, client.interactions_api())
}
