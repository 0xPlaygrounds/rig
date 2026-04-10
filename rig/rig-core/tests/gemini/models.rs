//! Gemini model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::gemini;

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn list_models_smoke() {
    let client = gemini::Client::from_env();
    let models = client
        .list_models()
        .await
        .expect("listing Gemini models should succeed");

    assert!(
        !models.is_empty(),
        "expected Gemini to return at least one model"
    );
}
