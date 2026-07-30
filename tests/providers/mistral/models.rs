//! Mistral model listing smoke test.

use rig::http_runtime::HttpRuntime;
use rig::providers::mistral;

use super::DEFAULT_MODEL;

#[tokio::test]
#[ignore = "requires MISTRAL_API_KEY"]
async fn list_models_smoke() {
    let cfg = mistral::functions::Config::from_env(DEFAULT_MODEL).expect("config should build");
    let rt = HttpRuntime::new();
    let models = match mistral::functions::list_models(&cfg, &rt).await {
        Ok(models) => models,
        Err(error) => {
            panic!("listing Mistral models should succeed\nDisplay: {error}\nDebug: {error:#?}")
        }
    };

    assert!(
        !models.is_empty(),
        "expected Mistral to return at least one model\nModel list: {models:#?}"
    );
}
