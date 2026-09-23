//! Request ids and `system_fingerprint` on Groq: see
//! `common/request_identity.rs`.

use rig::providers::groq;

use super::support::with_groq_cassette_result;
use crate::request_identity;

#[tokio::test]
async fn chat_completions() {
    const SCENARIO: &str = "request_identity_matrix/chat_completions";
    let slot = request_identity::Slot::default();
    let cell = slot.clone();
    with_groq_cassette_result(
        "request_identity_matrix/chat_completions",
        |client| async move {
            request_identity::run(
                cell,
                client.completion(groq::GPT_OSS_20B),
                client.completion("no-such-model"),
                None,
                |request| request,
            )
            .await;
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
    .expect("the cell runs");
    let observed = request_identity::take(&slot);
    request_identity::assert_recorded("groq", SCENARIO, "x-request-id", &observed);
}
