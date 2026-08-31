//! Moonshot model listing smoke test.
//!
//! Ignored rather than cassette-backed: no MOONSHOT_API_KEY was available when the
//! lister was added, so the endpoint could not be recorded. Moonshot documents
//! the OpenAI-style `{"object":"list","data":[…]}` envelope this decodes
//! (rig#2079).
use rig::client::ModelListingClient;
use rig::providers::moonshot;

#[tokio::test]
#[ignore = "requires MOONSHOT_API_KEY"]
async fn list_models_smoke() {
    let client = moonshot::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Moonshot models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected Moonshot to return at least one model\nModel list: {models:#?}"
    );
}
