//! Request ids on Anthropic: see `common/request_identity.rs`.

use super::super::support::with_anthropic_cassette;
use crate::request_identity;

#[tokio::test]
async fn messages() {
    const SCENARIO: &str = "request_identity_matrix/messages";
    let slot = request_identity::Slot::default();
    let cell = slot.clone();
    with_anthropic_cassette("request_identity_matrix/messages", |client| async move {
        request_identity::run(
            cell,
            client.completion("claude-haiku-4-5"),
            client.completion("claude-no-such-model"),
            None,
            |request| request,
        )
        .await
    })
    .await;
    let observed = request_identity::take(&slot);
    request_identity::assert_recorded("anthropic", SCENARIO, "request-id", &observed);
}
