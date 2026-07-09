//! ChatGPT cassette coverage for non-success Responses API status handling.

use axum::http;
use rig::client::CompletionClient;
use rig::completion::{CompletionError, CompletionModel};
use rig::providers::chatgpt;

use super::super::support::with_chatgpt_cassette;

#[tokio::test]
async fn nonstreaming_unauthorized_preserves_status_and_body() {
    with_chatgpt_cassette(
        "http_errors/nonstreaming_unauthorized_preserves_status_and_body",
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
    client: chatgpt::Client,
    expected_status: http::StatusCode,
    expected_message: &str,
) {
    let model = client.completion_model(chatgpt::GPT_5_4);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("non-success response should fail");

    assert!(matches!(&error, CompletionError::HttpError(_)));
    assert_eq!(error.provider_response_status(), Some(expected_status));
    let body = error
        .provider_response_body()
        .expect("provider response body should be preserved");
    assert!(
        body.contains(expected_message),
        "provider response body should include {expected_message:?}: {body}"
    );
}
