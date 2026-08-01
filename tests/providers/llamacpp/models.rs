//! llama.cpp model listing smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::openai;

use super::support;

#[tokio::test]
#[ignore = "requires a local llama.cpp OpenAI-compatible server"]
async fn list_models_smoke() {
    let cfg = support::completions_client().config(support::model_name());
    let rt = HttpRuntime::new();
    let models = match openai::functions::list_models(&cfg, &rt).await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing llama.cpp models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected llama.cpp to return at least one model\nModel list: {models:#?}"
    );
}
