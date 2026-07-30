//! Anthropic model listing smoke test.

use rig::providers::anthropic;
use rig::providers::anthropic::completion::CLAUDE_SONNET_4_6;

use super::super::support::with_anthropic_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_anthropic_cassette("models/list_models_smoke", |client| async move {
        // `list_models` needs a config; the model does not affect the request.
        let cfg = client.config(CLAUDE_SONNET_4_6);
        let http = client.http();
        let models = match anthropic::functions::list_models(&cfg, &http).await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing Anthropic models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected Anthropic to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
