//! Request ids on xAI: see `common/request_identity.rs`.

use rig::providers::xai;

use super::support::with_xai_cassette;
use crate::request_identity;

#[tokio::test]
async fn responses() {
    const SCENARIO: &str = "request_identity_matrix/responses";
    let slot = request_identity::Slot::default();
    let cell = slot.clone();
    with_xai_cassette("request_identity_matrix/responses", |client| async move {
        request_identity::run(
            cell,
            client.completion(xai::GROK_3_MINI),
            client.completion(xai::GROK_3_MINI),
            None,
            // xAI's unknown-model error quotes the account's team id.
            |request| request.temperature(7.0),
        )
        .await
    })
    .await;
    let observed = request_identity::take(&slot);
    request_identity::assert_recorded("xai", SCENARIO, "x-request-id", &observed);
}
