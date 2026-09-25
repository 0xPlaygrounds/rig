//! Contract-vs-reality (rig#2265): DeepSeek rides the OpenAI-compatible path
//! with the conservative `REQUEST_ID_HEADER = None` default.

use rig::providers::deepseek;

use super::support::with_deepseek_cassette;
use rig::completion::CompletionRequestBuilder;

#[tokio::test]
async fn blocking_contract_captures_none() {
    with_deepseek_cassette(
        "response_identity_edge/blocking_contract_captures_none",
        |client| async move {
            let model = rig::model(client.completion(deepseek::DEEPSEEK_V4_FLASH));
            let response = model
                .call(
                    CompletionRequestBuilder::new("Reply with exactly: identity probe").build(),
                    None,
                )
                .await
                .expect("completion should succeed");
            assert_eq!(response.provider_request_id, None);
        },
    )
    .await;
}
