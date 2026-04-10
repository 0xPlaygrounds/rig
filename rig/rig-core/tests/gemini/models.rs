//! Gemini model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::gemini;

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn list_models_smoke() {
    let client = gemini::Client::from_env();
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Gemini models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    println!("Gemini returned {} models", models.len());

    assert!(
        !models.is_empty(),
        "expected Gemini to return at least one model\nModel list: {models:#?}"
    );
}
