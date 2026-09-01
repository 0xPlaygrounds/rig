//! Contract-vs-reality (rig#2265): DeepSeek rides the OpenAI-compatible path
//! with the conservative `REQUEST_ID_HEADER = None` default.

use rig::completion::CompletionModel;
use rig::prelude::*;
use rig::providers::deepseek;

use super::support::with_deepseek_cassette;

#[tokio::test]
async fn blocking_contract_captures_none() {
    with_deepseek_cassette(
        "response_identity_edge/blocking_contract_captures_none",
        |client| async move {
            let model = client.completion_model(deepseek::DEEPSEEK_V4_FLASH);
            let response = model
                .completion_request("Reply with exactly: identity probe")
                .send()
                .await
                .expect("completion should succeed");
            assert_eq!(response.provider_request_id, None);
        },
    )
    .await;
}
