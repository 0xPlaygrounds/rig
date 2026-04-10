//! Mistral model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::mistral;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn list_models_smoke() {
    let client = mistral::Client::from_env();
    let models = client
        .list_models()
        .await
        .expect("listing Mistral models should succeed");

    assert!(
        !models.is_empty(),
        "expected Mistral to return at least one model"
    );
}
