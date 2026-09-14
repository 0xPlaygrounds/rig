//! Venice model listing smoke test.

use rig::client::ModelListingClient;

use super::super::support::with_venice_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_venice_cassette("model_listing/list_models_smoke", |client| async move {
        let models = client
            .list_models()
            .await
            .expect("listing Venice models should succeed");

        assert!(
            !models.is_empty(),
            "expected Venice to return at least one model\nModel list: {models:#?}"
        );
        assert!(
            models
                .iter()
                .any(|model| model.id == super::super::DEFAULT_MODEL),
            "expected the listing to include {}\nModel list: {models:#?}",
            super::super::DEFAULT_MODEL
        );
    })
    .await;
}
