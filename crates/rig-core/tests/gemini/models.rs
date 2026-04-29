//! Gemini model listing smoke test.

use rig_core::client::{ModelListingClient, ProviderClient};
use rig_core::providers::gemini;

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn list_models_smoke() {
    let client = gemini::Client::from_env().expect("client should build");
    let models = client
        .list_models()
        .await
        .expect("listing Gemini models should succeed");

    println!("Gemini returned {} models", models.len());

    assert!(
        !models.is_empty(),
        "expected Gemini to return at least one model\nModel list: {models:#?}"
    );
}
