//! Anthropic model listing smoke test.

use rig::client::ModelListingClient;

#[tokio::test]
async fn list_models_smoke() {
    let (cassette, client) =
        super::super::support::anthropic_cassette("models/list_models_smoke").await;
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Anthropic models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected Anthropic to return at least one model\nModel list: {models:#?}"
    );

    cassette.finish().await;
}
