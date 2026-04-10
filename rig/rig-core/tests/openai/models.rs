//! OpenAI model listing smoke test.

use rig::client::{ModelListingClient, ProviderClient};
use rig::providers::openai;

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn list_models_smoke() {
    let client = openai::Client::from_env();
    let models = client
        .list_models()
        .await
        .expect("listing OpenAI models should succeed");

    assert!(
        !models.is_empty(),
        "expected OpenAI to return at least one model"
    );
}
