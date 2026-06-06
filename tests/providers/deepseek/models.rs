//! DeepSeek model listing smoke test.
//!
//! Run with:
//! `cargo test -p rig --test deepseek list_models_smoke -- --ignored --nocapture`

use rig::client::ModelListingClient;

use super::support::with_deepseek_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_deepseek_cassette("models/list_models_smoke", |client| async move {
        let models = match client.list_models().await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing DeepSeek models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected DeepSeek to return at least one model\nModel list: {models:#?}"
        );

        assert!(
            models
                .iter()
                .any(|model| model.owned_by.as_deref() == Some("deepseek")),
            "expected at least one DeepSeek-owned model\nModel list: {models:#?}"
        );

        println!("DeepSeek returned {} models", models.len());
    })
    .await;
}
