//! Contract-vs-reality (rig#2265): Mistral rides the OpenAI-compatible path
//! with the conservative `REQUEST_ID_HEADER = None` default.

use anyhow::Result;
use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::mistral;

use super::support::with_mistral_cassette_result;

#[tokio::test]
async fn blocking_contract_captures_none() -> Result<()> {
    with_mistral_cassette_result(
        "response_identity_edge/blocking_contract_captures_none",
        |client| async move {
            let model = client.completion_model(mistral::MISTRAL_SMALL);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await?;
            anyhow::ensure!(
                response.provider_request_id.is_none(),
                "Mistral sends no request-id header and the contract is None; got {:?}",
                response.provider_request_id
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}
