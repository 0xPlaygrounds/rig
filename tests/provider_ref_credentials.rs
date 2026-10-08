//! A model from a provider reference, or connected through the catalog,
//! reads its credential from the environment the way the vendor's own
//! configuration does: a local `llama-server` needs none, Azure takes its
//! Entra token when no API key is set, sent the way that token is sent, and
//! an empty variable counts as unset.

use rig::catalog::Catalog;
use rig::providers::openai::{OpenAIConfig, wire::AZURE};
use rig::providers::registry::{ConnectError, ProviderConfig, ProviderRef};

/// Every reference built from the environment, in one test: the variables
/// are process-wide.
#[tokio::test]
async fn a_model_from_a_reference_reads_its_vendor_s_credential() -> anyhow::Result<()> {
    let server = httpmock::MockServer::start_async().await;
    let llama = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path_prefix("/llama/")
                .header_missing("authorization");
            then.status(500);
        })
        .await;
    let azure = server
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path_prefix("/azure/")
                .header("authorization", "Bearer entra-token")
                .header_missing("api-key");
            then.status(500);
        })
        .await;
    // SAFETY: this is the only test in this binary, and it reads the
    // environment only after writing it.
    unsafe {
        std::env::remove_var("LLAMACPP_API_KEY");
        std::env::set_var("LLAMACPP_API_BASE_URL", server.url("/llama/v1"));
        std::env::remove_var("AZURE_API_KEY");
        std::env::set_var("AZURE_TOKEN", "entra-token");
        std::env::set_var("DEEPSEEK_API_KEY", "");
    }

    // The replies are errors; what the servers matched is the evidence.
    let local = ProviderRef::parse("llamacpp/qwen3")?.completion_model()?;
    let _ = local.call("hi").await;
    let local = Catalog::builtin().connect("llamacpp/qwen3")?;
    let _ = local.call("hi").await;
    llama.assert_calls_async(2).await;

    let missing = Catalog::builtin()
        .connect("deepseek/deepseek-chat")
        .err()
        .ok_or_else(|| anyhow::anyhow!("an empty key is no key"))?;
    anyhow::ensure!(
        matches!(&missing, ConnectError::MissingKey { tried, .. } if tried == &["DEEPSEEK_API_KEY"]),
        "{missing}"
    );

    let configured = ProviderConfig::OpenAi(
        OpenAIConfig::with_key(&AZURE, "")
            .with_base_url(server.url("/azure"))
            .with_api_version("2024-10-21"),
    );
    let deployment = ProviderRef::configured(configured, "gpt-4o")?.completion_model()?;
    let _ = deployment.call("hi").await;
    azure.assert_async().await;
    Ok(())
}
