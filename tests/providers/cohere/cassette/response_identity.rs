//! Response identity metadata (rig#2265): Cohere reports no documented
//! request-id response header (its `x-debug-trace-id` is a debug trace
//! handle with unverified support semantics, deliberately not adopted), so
//! `provider_request_id` is `None` by design. This fixture is the recorded
//! proof of that absence.

use rig::completion::CompletionModel;
use rig::prelude::*;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};

#[tokio::test]
async fn nonstreaming_request_id_is_none_by_design() {
    with_cohere_cassette(
        "response_identity/nonstreaming_request_id_is_none_by_design",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .max_tokens(32)
                .send()
                .await
                .expect("completion should succeed");

            assert_eq!(
                response.provider_request_id, None,
                "Cohere has no adopted request-id header; None is the documented outcome"
            );
        },
    )
    .await;
}
