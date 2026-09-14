//! llama.cpp model listing smoke test.

use rig::client::ModelListingClient;

use super::super::cassette_support::*;

#[tokio::test]
async fn list_models_smoke() {
    with_llamacpp_cassette("models/list_models_smoke", |client| async move {
        let models = match client.list_models().await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing llama.cpp models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected llama.cpp to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
