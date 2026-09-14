//! MiniMax model listing smoke test.
//!
//! Ignored rather than cassette-backed: no MINIMAX_API_KEY was available when the
//! lister was added, so the endpoint could not be recorded. MiniMax documents
//! the OpenAI-style `{"object":"list","data":[…]}` envelope this decodes
//! (rig#2079).

use rig::client::DefaultTransportClient as _;
use rig::client::ModelListingClient;
use rig::providers::minimax;

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn list_models_smoke() {
    let client = minimax::Client::from_env().expect("client should build");
    let models = match client.list_models().await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing MiniMax models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected MiniMax to return at least one model\nModel list: {models:#?}"
    );
}
