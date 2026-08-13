//! Contract-vs-reality (rig#2265): Groq rides the OpenAI-compatible path with
//! the conservative `REQUEST_ID_HEADER = None` default — **and the recorded
//! fixture shows Groq actually sends `x-request-id`** (scrubbed to
//! `req_REDACTED_1` in the response headers). Rig deliberately captures
//! `None`: this cell is the live evidence for the maintainer question in
//! PR #2313 — should OpenAI-compatible providers that verifiably send the
//! header opt into `Some("x-request-id")`?

use anyhow::Result;
use rig::completion::CompletionModel;
use rig::prelude::*;

use super::support::with_groq_cassette_result;

#[tokio::test]
async fn blocking_contract_captures_none() -> Result<()> {
    with_groq_cassette_result(
        "response_identity_edge/blocking_contract_captures_none",
        |client| async move {
            let model = client.completion_model("llama-3.3-70b-versatile");
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await?;
            anyhow::ensure!(
                response.provider_request_id.is_none(),
                "the header is on the wire (see the fixture) but the compat \
                 contract deliberately does not capture it; got {:?}",
                response.provider_request_id
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
