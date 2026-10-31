use super::*;
use crate::completion::CompletionRequest;
use crate::effect::EffectId;
use crate::providers::chatgpt::auth::{
    AuthSource, Authenticator, DeviceCodeHandler, MemoryTokenStore, TokenStore,
};
use crate::test_utils::{MockHttpResponse, SequencedHttpClient};
use http::StatusCode;
use std::time::{Duration, SystemTime};

/// A model call whose expired ChatGPT credential the token endpoint refuses
/// to refresh with `reply`.
async fn refreshed_with(reply: MockHttpResponse) -> anyhow::Result<ErrorReport> {
    let store = Arc::new(MemoryTokenStore::default());
    store
        .save(
            chatgpt::PROVIDER_NAME,
            r#"{"refresh_token":"synthetic-refresh","expires_at":0}"#,
        )
        .await
        .map_err(anyhow::Error::from_boxed)?;
    let auth = Authenticator::new(
        AuthSource::OAuth,
        store,
        DeviceCodeHandler::default(),
        false,
    );
    let http = DynHttpClient::new(SequencedHttpClient::new([reply]));
    let spec = Catalog::builtin().resolve("chatgpt/gpt-5.5")?.spec.clone();
    let model = SignedInModel::new(spec, auth, http)?;
    let kind = EffectKind::Completion {
        request: Arc::new(CompletionRequest::new("hi")),
        stream: false,
    };
    let outcome = model
        .serve(kind, Dispatch::new(EffectId::from_raw(0), false))
        .await
        .into_outcome()
        .await;
    outcome
        .err()
        .ok_or_else(|| anyhow::anyhow!("the call had no credential"))
}

/// A refresh the server could not serve now is retried, waiting as long as
/// it asks; one it refused is final and asks to sign in again.
#[tokio::test]
async fn a_failed_refresh_keeps_the_class_of_its_reply() -> anyhow::Result<()> {
    let mut wait = http::HeaderMap::new();
    wait.insert(http::header::RETRY_AFTER, "7".parse()?);
    let refused = r#"{"error":"invalid_request","error_description":"no"}"#;
    for (reply, retryable, asked) in [
        (
            MockHttpResponse::error_with_headers(StatusCode::TOO_MANY_REQUESTS, "slow down", wait),
            true,
            Some(Duration::from_secs(7)),
        ),
        (
            MockHttpResponse::error(StatusCode::SERVICE_UNAVAILABLE, "down"),
            true,
            None,
        ),
        (
            MockHttpResponse::error(StatusCode::BAD_REQUEST, refused),
            false,
            None,
        ),
    ] {
        let report = refreshed_with(reply).await?;
        let message = &report.message;
        anyhow::ensure!(report.retryable == retryable, "{report:?}");
        anyhow::ensure!(report.retry_after(SystemTime::now()) == asked, "{report:?}");
        anyhow::ensure!(
            message.contains("ChatGPT token refresh failed"),
            "{message}"
        );
        anyhow::ensure!(message.ends_with("sign in again") != retryable, "{message}");
    }
    Ok(())
}
