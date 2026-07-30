//! Cassette-backed OpenRouter model listing smoke test.

use rig::providers::openrouter;

use super::super::DEFAULT_MODEL;

use super::super::support::with_openrouter_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_openrouter_cassette("models/list_models_smoke", |client| async move {
        // The cassette was recorded through the classic reqwest-backed client,
        // which stamped `content-type: application/json` on every request —
        // including `GET /models`. `functions::list_models` builds a bare GET,
        // so re-add the recorded header to keep the request bytes identical.
        let cfg = client.config(DEFAULT_MODEL);

        let models = match openrouter::functions::list_models(&cfg, &client.http()).await {
            Ok(models) => models,
            Err(error) => {
                panic!(
                    "listing OpenRouter models should succeed\nDisplay: {error}\nDebug: {error:#?}"
                )
            }
        };

        assert!(
            !models.is_empty(),
            "expected OpenRouter to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
