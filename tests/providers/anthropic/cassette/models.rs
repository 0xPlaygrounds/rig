//! Anthropic model listing smoke test.

use rig::client::ModelListingClient;

use super::super::support::with_anthropic_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_anthropic_cassette("models/list_models_smoke", |client| async move {
        let models = match client.list_models().await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing Anthropic models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected Anthropic to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
