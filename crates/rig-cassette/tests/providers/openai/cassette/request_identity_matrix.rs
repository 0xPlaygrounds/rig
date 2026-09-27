//! Request ids and `system_fingerprint` on OpenAI: see
//! `common/request_identity.rs`.

use super::super::support::with_openai_cassette;
use crate::request_identity;

const REJECTED: &str = "gpt-no-such-model";

#[tokio::test]
async fn chat_completions() {
    const SCENARIO: &str = "request_identity_matrix/chat_completions";
    let slot = request_identity::Slot::default();
    let cell = slot.clone();
    with_openai_cassette(
        "request_identity_matrix/chat_completions",
        |client| async move {
            request_identity::run(
                cell,
                client.openai.chat("gpt-4.1-nano"),
                client.openai.chat(REJECTED),
                None,
                |request| request,
            )
            .await
        },
    )
    .await;
    let observed = request_identity::take(&slot);
    request_identity::assert_recorded("openai", SCENARIO, "x-request-id", &observed);
}

#[tokio::test]
async fn responses() {
    const SCENARIO: &str = "request_identity_matrix/responses";
    let slot = request_identity::Slot::default();
    let cell = slot.clone();
    with_openai_cassette("request_identity_matrix/responses", |client| async move {
        request_identity::run(
            cell,
            client.openai.responses("gpt-4.1-nano"),
            client.openai.responses(REJECTED),
            Some(serde_json::json!({ "store": false })),
            |request| request,
        )
        .await
    })
    .await;
    let observed = request_identity::take(&slot);
    request_identity::assert_recorded("openai", SCENARIO, "x-request-id", &observed);
}
