//! OpenAI model listing smoke test.

use rig::providers::openai;

use super::super::support::with_openai_cassette;

#[tokio::test]
async fn list_models_smoke() {
    with_openai_cassette("models/list_models_smoke", |client| async move {
        // `list_models` only reads the base URL and credentials off the config;
        // the model field does not affect the recorded request bytes.
        let cfg = client.completions_config(openai::GPT_4O);
        let models = match openai::functions::list_models(&cfg, &client.http()).await {
            Ok(models) => models,
            Err(error) => {
                panic!("listing OpenAI models should succeed\nDisplay: {error}\nDebug: {error:#?}")
            }
        };

        assert!(
            !models.is_empty(),
            "expected OpenAI to return at least one model\nModel list: {models:#?}"
        );
    })
    .await;
}
