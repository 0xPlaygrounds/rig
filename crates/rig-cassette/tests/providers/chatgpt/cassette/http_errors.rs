//! ChatGPT cassette coverage for non-success Responses API status handling.

use axum::http;
use rig::error::ProviderError;
use rig::providers::chatgpt;
use rig::providers::openai::OpenAI;
use rig::wire::Wire as _;

use super::super::support::with_chatgpt_cassette;
use rig::completion::CompletionRequestBuilder;

#[tokio::test]
async fn nonstreaming_unauthorized_preserves_status_and_body() {
    with_chatgpt_cassette(
        crate::cassettes::CassetteSpec::new(
            "http_errors/nonstreaming_unauthorized_preserves_status_and_body",
        )
        .expects_account_failure(crate::cassettes::AccountFailure::Auth),
        |client| async move {
            assert_nonstreaming_http_error(
                client,
                http::StatusCode::UNAUTHORIZED,
                "authentication token",
            )
            .await;
        },
    )
    .await;
}

async fn assert_nonstreaming_http_error(
    client: OpenAI,
    expected_status: http::StatusCode,
    expected_message: &str,
) {
    let model = client.completion(chatgpt::GPT_5_4).on(rig::transport());
    let request = CompletionRequestBuilder::new("hello").build();

    let error = model
        .call(request)
        .await
        .expect_err("non-success response should fail");

    assert!(matches!(&error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_response_status(), Some(expected_status));
    let body = error
        .provider_response_body()
        .expect("provider response body should be preserved");
    assert!(
        body.contains(expected_message),
        "provider response body should include {expected_message:?}: {body}"
    );
}
