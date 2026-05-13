//! Gemini model listing smoke test.

use rig::client::ModelListingClient;

#[tokio::test]
async fn list_models_smoke() {
    let (cassette, client) =
        super::super::support::gemini_cassette("models/list_models_smoke").await;
    let models = client
        .list_models()
        .await
        .expect("listing Gemini models should succeed");

    println!("Gemini returned {} models", models.len());

    assert!(
        !models.is_empty(),
        "expected Gemini to return at least one model\nModel list: {models:#?}"
    );

    cassette.finish().await;
}
