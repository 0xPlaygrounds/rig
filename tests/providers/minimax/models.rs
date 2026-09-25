//! MiniMax model listing smoke test.
//!
//! Ignored rather than cassette-backed: no MINIMAX_API_KEY was available when the
//! lister was added, so the endpoint could not be recorded. MiniMax documents
//! the OpenAI-style `{"object":"list","data":[…]}` envelope this decodes
//! (rig#2079).

use rig::providers::openai::wire::{self as openai_wire, OpenAI};
use rig_test_support::endpoint::Endpoint;

#[tokio::test]
#[ignore = "requires MINIMAX_API_KEY"]
async fn list_models_smoke() {
    let client = Endpoint::new(
        OpenAI::from_env_with(&openai_wire::MINIMAX).expect("MINIMAX_API_KEY should be set"),
        rig::rig_reqwest::shared(),
    );
    let models = match client.models().call((), None).await {
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
